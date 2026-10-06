"""Chunked HDF5 grids, physical XY bilinear queries and linear physical Z queries.

Arrays are [row_y, column_x, embedding]. Origin is the first grid CENTER.
Only one coordinate frame and one section per physical Z are allowed per store.
Queries return tensors on the input tensor's device; no encoder/image access.
"""

from collections import OrderedDict
import hashlib
import json
import uuid
import h5py
import numpy as np
import torch


class FeatureStore:
    def __init__(self, path, mode="r", cache_mb=512):
        if mode not in ("r", "a", "x"):
            raise ValueError("Use r/a/x; destructive overwrite is not supported")
        self.path = str(path)
        self.cache_bytes = int(cache_mb * 1024**2)
        self._cache = OrderedDict()
        if mode != "r":
            with h5py.File(self.path, mode) as f:
                f.require_group("sections")
                if "revision" not in f.attrs:
                    f.attrs["revision"] = str(uuid.uuid4())
                f.attrs["format"] = "deepspatial-uni2-grid-v1"
        self.refresh()

    def refresh(self):
        self._cache.clear()
        self._spatial_indices = {}
        with h5py.File(self.path, "r") as f:
            self.revision = str(f.attrs["revision"])
            self.metadata = {
                str(g.attrs["section_id"]): json.loads(g.attrs["metadata"])
                for key, g in f["sections"].items()
                if not key.startswith("_pending_")
            }
        self.sections = sorted(self.metadata, key=lambda s: self.metadata[s]["z_um"])
        self.zs = np.array([self.metadata[s]["z_um"] for s in self.sections])
        self.feature_dim = (
            self.metadata[self.sections[0]]["shape"][-1] if self.sections else None
        )

    @staticmethod
    def _key(sid):
        return hashlib.sha256(str(sid).encode()).hexdigest()

    def add_section(
        self,
        section_id,
        features,
        *,
        z_um,
        origin_um,
        spacing_um,
        patch_size_um,
        mpp,
        coordinate_frame,
        valid_mask=None,
        provenance=None,
    ):
        f = np.asarray(features)
        if f.ndim != 3 or min(f.shape) < 1 or not np.isfinite(f).all():
            raise ValueError("Features must be finite [H,W,D]")
        for key, val in [("origin", origin_um), ("spacing", spacing_um), ("mpp", mpp)]:
            if (
                len(val) != 2
                or not np.isfinite(val).all()
                or (key != "origin" and min(val) <= 0)
            ):
                raise ValueError(f"Invalid {key}")
        if (
            not np.isfinite(z_um)
            or not np.isfinite(patch_size_um)
            or patch_size_um <= 0
            or not coordinate_frame
        ):
            raise ValueError(
                "Finite physical Z, positive FOV and explicit frame required"
            )
        sid = str(section_id)
        if sid in self.metadata or z_um in self.zs:
            raise ValueError(
                "Duplicate section ID or physical Z; use a separate series store"
            )
        if self.sections:
            ref = self.metadata[self.sections[0]]
            if (
                ref["coordinate_frame"] != coordinate_frame
                or self.feature_dim != f.shape[-1]
            ):
                raise ValueError("Feature dimensions and coordinate frame must agree")
            if ref["patch_size_um"] != float(patch_size_um):
                raise ValueError("Physical patch FOV must be constant across sections")
        mask = (
            np.ones(f.shape[:2], bool)
            if valid_mask is None
            else np.asarray(valid_mask, bool)
        )
        if mask.shape != f.shape[:2]:
            raise ValueError("valid_mask must have shape [H,W]")
        meta = dict(
            z_um=float(z_um),
            origin_um=list(map(float, origin_um)),
            spacing_um=list(map(float, spacing_um)),
            patch_size_um=float(patch_size_um),
            mpp=list(map(float, mpp)),
            coordinate_frame=coordinate_frame,
            shape=list(f.shape),
            provenance=provenance or {},
        )
        with h5py.File(self.path, "a") as out:
            pending = "_pending_" + str(uuid.uuid4())
            g = out["sections"].create_group(pending)
            g.attrs["section_id"] = sid
            g.attrs["metadata"] = json.dumps(meta)
            g.create_dataset(
                "features",
                data=f.astype("float32"),
                chunks=(min(16, f.shape[0]), min(16, f.shape[1]), f.shape[2]),
                compression="lzf",
            )
            g.create_dataset("valid", data=mask, chunks=True, compression="lzf")
            out.flush()
            out["sections"].move(pending, self._key(sid))
            out.attrs["revision"] = str(uuid.uuid4())
        self.refresh()

    def _corners(self, sid, iy, ix):
        """Lazy gather, caching small planes; large planes are read by row spans."""
        if sid in self._cache:
            f, m = self._cache.pop(sid)
            self._cache[sid] = (f, m)
            return f[iy, ix], m[iy, ix]
        with h5py.File(self.path, "r") as file:
            g = file["sections"][self._key(sid)]
            d = g["features"]
            valid = g["valid"]
            size = int(np.prod(d.shape)) * 4 + int(np.prod(d.shape[:2]))
            if size <= self.cache_bytes:
                while (
                    self._cache
                    and sum(a.nbytes + b.nbytes for a, b in self._cache.values()) + size
                    > self.cache_bytes
                ):
                    self._cache.popitem(last=False)
                f, m = d[:], valid[:]
                self._cache[sid] = (f, m)
                return f[iy, ix], m[iy, ix]
            out = np.empty((len(iy), d.shape[-1]), np.float32)
            ok = np.empty(len(iy), bool)
            for row in np.unique(iy):
                idx = np.flatnonzero(iy == row)
                cols, inverse = np.unique(ix[idx], return_inverse=True)
                out[idx] = d[int(row), cols, :][inverse]
                ok[idx] = valid[int(row), cols][inverse]
            return out, ok

    def _nearest_valid(self, sid, xy, maximum_distance_um):
        """Return nearest valid grid features and a bounded acceptance mask."""
        from scipy.spatial import cKDTree

        if sid not in self._spatial_indices:
            metadata = self.metadata[sid]
            with h5py.File(self.path, "r") as file:
                valid = file["sections"][self._key(sid)]["valid"][:]
            iy, ix = np.nonzero(valid)
            coordinates = np.column_stack(
                [
                    metadata["origin_um"][0] + ix * metadata["spacing_um"][0],
                    metadata["origin_um"][1] + iy * metadata["spacing_um"][1],
                ]
            )
            self._spatial_indices[sid] = (cKDTree(coordinates), iy, ix)
        tree, iy, ix = self._spatial_indices[sid]
        distance, nearest = tree.query(np.asarray(xy, dtype=float), k=1)
        accepted = np.asarray(distance) <= float(maximum_distance_um)
        values = np.zeros((len(xy), self.feature_dim), dtype=np.float32)
        if accepted.any():
            selected = np.asarray(nearest)[accepted]
            gathered, _ = self._corners(sid, iy[selected], ix[selected])
            values[accepted] = gathered
        return values, accepted

    def get_feature(
        self,
        section_id,
        xy,
        *,
        return_valid=False,
        return_quality=False,
        nearest_max_distance_um=None,
        minimum_valid_weight=1e-6,
    ):
        """Query masked bilinear morphology features in physical XY.

        Quality codes are 0=unavailable, 1=full bilinear, 2=partial masked
        bilinear and 3=bounded nearest-valid fallback.
        """
        xy = torch.as_tensor(xy)
        if not xy.is_floating_point():
            xy = xy.float()
        if xy.ndim != 2 or xy.shape[1] != 2 or not torch.isfinite(xy).all():
            raise ValueError("xy must be finite [N,2]")
        sid = str(section_id)
        m = self.metadata[sid]
        h, w, _ = m["shape"]
        uv = (xy.detach().cpu().numpy() - m["origin_um"]) / m["spacing_um"]
        inside = (
            (uv[:, 0] >= -1e-6)
            & (uv[:, 0] <= w - 1 + 1e-6)
            & (uv[:, 1] >= -1e-6)
            & (uv[:, 1] <= h - 1 + 1e-6)
        )
        uv = np.clip(uv, [0, 0], [w - 1, h - 1])
        lo = np.floor(uv).astype(int)
        hi = np.minimum(lo + 1, [w - 1, h - 1])
        frac = uv - lo
        values = np.zeros((len(xy), self.feature_dim), np.float32)
        valid_weight = np.zeros(len(xy), dtype=np.float64)
        all_active_corners_valid = np.ones(len(xy), dtype=bool)
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            ix = hi[:, 0] if a else lo[:, 0]
            iy = hi[:, 1] if b else lo[:, 1]
            weight = (frac[:, 0] if a else 1 - frac[:, 0]) * (
                frac[:, 1] if b else 1 - frac[:, 1]
            )
            v, valid = self._corners(sid, iy, ix)
            effective = weight * valid
            values += v * effective[:, None]
            valid_weight += effective
            all_active_corners_valid &= valid | (weight < 1e-8)
        ok = (
            inside
            & (valid_weight >= float(minimum_valid_weight))
            & np.isfinite(valid_weight)
        )
        values[ok] /= valid_weight[ok, None]
        ok &= np.linalg.norm(values, axis=1) > 1e-10
        quality = np.zeros(len(xy), dtype=np.uint8)
        quality[ok & all_active_corners_valid] = 1
        quality[ok & ~all_active_corners_valid] = 2

        maximum = (
            1.5 * max(m["spacing_um"])
            if nearest_max_distance_um is None
            else float(nearest_max_distance_um)
        )
        missing = ~ok
        if maximum > 0 and missing.any():
            nearest_values, nearest_ok = self._nearest_valid(
                sid, xy.detach().cpu().numpy()[missing], maximum
            )
            recovered = np.flatnonzero(missing)[nearest_ok]
            values[recovered] = nearest_values[nearest_ok]
            ok[recovered] = True
            quality[recovered] = 3
        values[~ok] = 0
        result = torch.as_tensor(values, device=xy.device, dtype=xy.dtype)
        mask = torch.as_tensor(ok, device=xy.device)
        methods = torch.as_tensor(quality, device=xy.device)
        if not return_valid and not return_quality and not ok.all():
            raise ValueError(
                f"{sid}: {int((~ok).sum())} feature queries outside valid tissue/grid"
            )
        if return_quality:
            return result, mask, methods
        return (result, mask) if return_valid else result

    def query(self, xy, z_um, *, return_valid=False):
        xy = torch.as_tensor(xy)
        if not xy.is_floating_point():
            xy = xy.float()
        z = np.broadcast_to(
            torch.as_tensor(z_um).detach().cpu().numpy().reshape(-1), (len(xy),)
        )
        if (
            not len(self.zs)
            or not np.isfinite(z).all()
            or (z < self.zs[0] - 1e-5).any()
            or (z > self.zs[-1] + 1e-5).any()
        ):
            raise ValueError("Physical Z outside feature coverage")
        z = np.clip(z, self.zs[0], self.zs[-1])
        hi = np.searchsorted(self.zs, z, side="left").clip(0, len(self.zs) - 1)
        lo = np.maximum(hi - 1, 0)
        lo[np.isclose(z, self.zs[hi], atol=1e-7, rtol=0)] = hi[
            np.isclose(z, self.zs[hi], atol=1e-7, rtol=0)
        ]
        out = torch.empty((len(xy), self.feature_dim), device=xy.device, dtype=xy.dtype)
        valid = torch.ones(len(xy), device=xy.device, dtype=torch.bool)
        for l, h in np.unique(np.stack([lo, hi], 1), axis=0):
            ids = np.flatnonzero((lo == l) & (hi == h))
            idx = torch.as_tensor(ids, device=xy.device)
            f0, m0 = self.get_feature(self.sections[l], xy[idx], return_valid=True)
            if l == h:
                out[idx] = f0
                valid[idx] = m0
                continue
            f1, m1 = self.get_feature(self.sections[h], xy[idx], return_valid=True)
            a = torch.as_tensor(
                (z[ids] - self.zs[l]) / (self.zs[h] - self.zs[l]),
                device=xy.device,
                dtype=xy.dtype,
            )[:, None]
            out[idx] = (1 - a) * f0 + a * f1
            valid[idx] = m0 & m1
        if not return_valid and not valid.all():
            raise ValueError("Invalid XY feature query at interpolated Z")
        return (out, valid) if return_valid else out
