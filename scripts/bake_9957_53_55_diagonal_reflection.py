"""Bake the reviewed 53->55 diagonal reflection into a direct STalign edge.

The input is an isolated direct-STalign candidate.  The reflection is applied
in the target (section-53) physical canvas, not merely to a preview PNG:

    T_new = R_diag @ T_stalign

The saved STalign ``A`` values, masks, images, matching weights, and summary
are all updated in a new output directory.  The existing candidate is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_SRC = PROJECT_ROOT.parent / "DeepSpatial_v2" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.serial_registration.partial_overlap_qc import (  # noqa: E402
    compute_partial_overlap_metrics,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    _apply_residual_to_target_image,
    _apply_residual_to_target_mask,
)
from deepspatial_v2.serial_registration.stalign_types import RasterizedSection  # noqa: E402


def _xy_to_rc(matrix_xy: np.ndarray) -> np.ndarray:
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return swap @ np.asarray(matrix_xy, dtype=float) @ swap


def _diagonal_reflection(origin_um: tuple[float, float]) -> np.ndarray:
    x0, y0 = map(float, origin_um)
    # Reflection about the canvas line through its upper-left corner with
    # slope +1: (x-x0, y-y0) -> (y-y0, x-x0).
    return np.array(
        [[0.0, 1.0, x0 - y0], [1.0, 0.0, y0 - x0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _load_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _save_map(path: Path, value: dict[str, np.ndarray], reflection_rc: np.ndarray) -> None:
    updated = dict(value)
    if "A" not in updated:
        raise ValueError(f"STalign map has no A: {path}")
    updated["A"] = reflection_rc @ np.asarray(updated["A"], dtype=float)
    if "WM" in updated:
        updated["WM"] = np.asarray(updated["WM"], dtype=np.float32)
    np.savez_compressed(path, **updated)


def _rgb_to_chw(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {rgb.shape}")
    return np.moveaxis(rgb / 255.0, -1, 0)


def _chw_to_rgb(value: np.ndarray) -> np.ndarray:
    chw = np.asarray(value, dtype=np.float32)
    rgb = np.moveaxis(np.clip(chw, 0.0, 1.0), 0, -1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _he_redgreen(fixed_rgb: np.ndarray, moving_rgb: np.ndarray) -> np.ndarray:
    fixed_dark = np.clip(1.0 - fixed_rgb.astype(np.float32).mean(axis=2) / 255.0, 0.0, 1.0)
    moving_dark = np.clip(1.0 - moving_rgb.astype(np.float32).mean(axis=2) / 255.0, 0.0, 1.0)
    result = np.zeros((*fixed_dark.shape, 3), dtype=np.float32)
    result[..., 0] = fixed_dark
    result[..., 1] = moving_dark
    return np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)


def _crop_and_upscale(path: Path, threshold: int = 10, scale: int = 3) -> Path:
    image = Image.open(path).convert("RGB")
    value = np.asarray(image)
    visible = value.max(axis=2) > int(threshold)
    ys, xs = np.where(visible)
    if len(xs):
        pad = 8
        box = (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
        image = image.crop(box)
    output = path.with_name(path.stem + "_cropped.png")
    image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST).save(output)
    return output


def bake(input_root: Path, output_root: Path) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    pair = input_root / "pair-53-55"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(input_root, output_root)
    out_pair = output_root / "pair-53-55"

    summary = json.loads((input_root / "direct_stalign_summary.json").read_text(encoding="utf-8"))
    origin = tuple(float(x) for x in summary["canvas_origin_um"])
    shape_rc = tuple(int(x) for x in summary["canvas_shape_rc"])
    dx_um = float(summary["dx_um"])
    reflection_xy = _diagonal_reflection(origin)
    reflection_rc = _xy_to_rc(reflection_xy)

    target_y = origin[1] + dx_um * np.arange(shape_rc[0], dtype=float)
    target_x = origin[0] + dx_um * np.arange(shape_rc[1], dtype=float)
    xx, yy = np.meshgrid(target_x, target_y, indexing="xy")
    target_grid = np.stack([xx, yy], axis=0)
    target = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=53,
        section_uid="9957__g0__section-53",
        image=np.zeros((3, *shape_rc), dtype=np.float32),
        grid_um=target_grid,
        mask=np.zeros(shape_rc, dtype=bool),
        origin_um=origin,
        dx_um=dx_um,
    )

    fixed_rgb = np.asarray(Image.open(pair / "fixed_section53_v6_frame.png").convert("RGB"))
    old_moving_rgb = np.asarray(Image.open(pair / "moving_section55_after_direct_stalign.png").convert("RGB"))
    with np.load(pair / "direct_stalign_masks.npz", allow_pickle=False) as data:
        fixed_mask = np.asarray(data["fixed_mask"], dtype=bool)
        old_moving_mask = np.asarray(data["moving_mask_after"], dtype=bool)
        moving_before = np.asarray(data["moving_mask_before"], dtype=bool)
    if fixed_mask.shape != shape_rc or old_moving_mask.shape != shape_rc:
        raise ValueError("saved mask shape does not match the target canvas")

    new_moving_mask = _apply_residual_to_target_mask(old_moving_mask, target, reflection_xy)
    new_moving_rgb = _chw_to_rgb(
        _apply_residual_to_target_image(_rgb_to_chw(old_moving_rgb), target, reflection_xy)
    )
    new_moving_rgb[~new_moving_mask] = 255

    Image.fromarray(new_moving_rgb).save(out_pair / "moving_section55_after_direct_stalign_diagonal_reflection.png")
    Image.fromarray(new_moving_rgb).save(out_pair / "moving_section55_after_direct_stalign.png")
    Image.fromarray(_he_redgreen(fixed_rgb, new_moving_rgb)).save(
        out_pair / "overlay_after_he_red53_green55.png"
    )
    Image.fromarray(
        np.where(
            (fixed_mask & new_moving_mask)[..., None],
            np.array([255, 220, 120], dtype=np.uint8),
            np.where(
                fixed_mask[..., None],
                np.array([230, 0, 0], dtype=np.uint8),
                np.where(new_moving_mask[..., None], np.array([0, 220, 0], dtype=np.uint8), 0),
            ),
        ).astype(np.uint8)
    ).save(out_pair / "overlay_after_red53_green55.png")

    np.savez_compressed(
        out_pair / "direct_stalign_masks.npz",
        fixed_mask=fixed_mask,
        moving_mask_before=moving_before,
        moving_mask_after=new_moving_mask,
    )

    forward = _load_map(pair / "direct_stalign_forward_map.npz")
    reverse = _load_map(pair / "direct_stalign_reverse_map.npz")
    _save_map(out_pair / "direct_stalign_forward_map.npz", forward, reflection_rc)
    _save_map(out_pair / "direct_stalign_reverse_map.npz", reverse, reflection_rc)
    old_affine = np.asarray(summary["direct_stalign_affine"], dtype=float)
    new_affine = reflection_xy @ old_affine
    np.savez_compressed(
        out_pair / "direct_stalign_affine.npz",
        affine=new_affine,
        initial_affine=np.eye(3, dtype=float),
        origin_um=np.asarray(origin, dtype=float),
        dx_um=np.asarray([dx_um], dtype=float),
        shape_rc=np.asarray(shape_rc, dtype=np.int64),
        diagonal_reflection_xy=reflection_xy,
    )
    if "WM" in forward:
        wm = np.asarray(forward["WM"], dtype=np.float32)
        wm_reflected = _apply_residual_to_target_image(wm[None, ...], target, reflection_xy)[0]
        np.savez_compressed(
            out_pair / "direct_stalign_matching_weight_baked.npz",
            matching_weight=wm_reflected,
        )

    baseline = summary.get("baseline_metrics", {})
    candidate = compute_partial_overlap_metrics(
        fixed_mask, new_moving_mask, dx_um, trim_fraction=0.2, tolerance_um=20.0
    )
    summary.update(
        {
            "status": "direct_stalign_completed_diagonal_reflection_baked_preview",
            "parent_candidate": str(input_root.resolve()),
            "post_stalign_target_reflection": "upper-left_to_lower-right_diagonal",
            "post_stalign_target_reflection_baked": True,
            "diagonal_reflection_xy": reflection_xy.tolist(),
            "direct_stalign_affine": new_affine.tolist(),
            "baseline_metrics": baseline,
            "candidate_metrics_before_diagonal_reflection": summary.get("candidate_metrics", {}),
            "candidate_metrics": candidate,
            "map_semantics": "forward/reverse map A values already include the target-frame diagonal reflection",
            "downstream_propagation": "not yet applied to sections >=55; preview only",
            "he_overlay": "red=section-53 optical density; green=section-55 after baked reflection; yellow=overlap",
        }
    )
    (output_root / "direct_stalign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    crop = _crop_and_upscale(out_pair / "overlay_after_he_red53_green55.png")
    summary["he_overlay_cropped"] = str(crop.resolve())
    (output_root / "direct_stalign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = bake(args.input_root.resolve(), args.output_root.resolve())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_root": str(args.output_root.resolve()),
                "diagonal_reflection_xy": summary["diagonal_reflection_xy"],
                "direct_stalign_affine": summary["direct_stalign_affine"],
                "candidate_metrics": summary["candidate_metrics"],
                "he_overlay_cropped": summary["he_overlay_cropped"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
