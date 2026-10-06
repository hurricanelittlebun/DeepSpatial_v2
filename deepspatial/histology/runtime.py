"""Physical/normalized coordinate bridge, shared by preprocessing, FM and ODE."""

import numpy as np
import torch
from .feature_store import FeatureStore
from .path import MorphologyPathCache


class HistologyRuntime:
    def __init__(self, config, spatial_stats):
        self.config = config
        self.stats = spatial_stats
        self.path_ids = []
        self.store = FeatureStore(config.feature_path, cache_mb=config.feature_cache_mb)
        self.nucleus_store = None
        if config.use_nucleus_path:
            self.nucleus_store = FeatureStore(
                config.nucleus_feature_path, cache_mb=config.feature_cache_mb
            )
            if self.nucleus_store.sections != self.store.sections:
                raise ValueError("Nucleus and UNI2 stores must contain the same sections")
            for sid in self.store.sections:
                left = self.store.metadata[sid]
                right = self.nucleus_store.metadata[sid]
                if (
                    left["coordinate_frame"] != right["coordinate_frame"]
                    or left["shape"][:2] != right["shape"][:2]
                    or not np.allclose(left["origin_um"], right["origin_um"])
                    or not np.allclose(left["spacing_um"], right["spacing_um"])
                    or not np.isclose(left["z_um"], right["z_um"])
                ):
                    raise ValueError(f"Nucleus store grid mismatch for section {sid}")
        if not self.store.sections:
            raise ValueError("Empty histology feature store")
        if any(
            m["coordinate_frame"] != config.coordinate_frame
            for m in self.store.metadata.values()
        ):
            raise ValueError(
                "Feature coordinate_frame differs from requested registered series"
            )
        self.paths = (
            MorphologyPathCache(
                config.path_cache,
                self.store,
                candidate_radius_um=config.candidate_radius_um,
                candidate_spacing_um=config.candidate_spacing_um,
                move_weight=config.move_weight,
                morphology_weight=config.morphology_weight,
                nucleus_store=self.nucleus_store,
                nucleus_weight=config.nucleus_weight,
                max_candidates=config.max_candidates,
                allow_linear_fallback=config.allow_linear_fallback,
            )
            if config.use_morphology_path
            else None
        )

    def xy_um(self, x):
        offset = [self.stats["x_min"], self.stats["y_min"]]
        scale = [self.stats["x_range"], self.stats["y_range"]]
        if torch.is_tensor(x):
            return x * x.new_tensor(scale) + x.new_tensor(offset)
        return x * np.array(scale) + np.array(offset)

    def features(self, x, z):
        z_um = z * self.stats["z_range"] + self.stats["z_min"]
        features, _ = self.store.query(self.xy_um(x), z_um, return_valid=True)
        return features

    def plan(self, indices, t):
        ids = [self.path_ids[i] for i in indices.detach().cpu().reshape(-1).tolist()]
        x, v = self.paths.evaluate(ids, t)
        scale = x.new_tensor([self.stats["x_range"], self.stats["y_range"]])
        origin = x.new_tensor([self.stats["x_min"], self.stats["y_min"]])
        return (x - origin) / scale, v / scale

    def section(self, adata):
        key = self.config.section_key
        if key not in adata.obs or adata.obs[key].nunique() != 1:
            raise ValueError(f"Each anchor needs exactly one obs[{key!r}]")
        sid = str(adata.obs[key].iloc[0])
        if sid not in self.store.metadata:
            raise ValueError(f"Missing anchor H&E feature plane: {sid}")
        z = (
            float(adata.obs["z_norm"].iloc[0]) * self.stats["z_range"]
            + self.stats["z_min"]
        )
        if not np.isclose(z, self.store.metadata[sid]["z_um"], rtol=0, atol=1e-4):
            raise ValueError(f"Physical Z mismatch for {sid}")
        return sid
