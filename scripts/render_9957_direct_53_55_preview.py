"""Render a readable H&E optical-density red/green preview for a direct edge."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _optical_density(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32) / 255.0
    # The saved morphology image is RGB on a white background.  Mean darkness
    # is used only for visualization; registration itself used the morphology
    # channels and the saved STalign map.
    return np.clip(1.0 - value.mean(axis=2), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, required=True)
    args = parser.parse_args()
    pair_dir = args.pair_dir.resolve()
    fixed = np.asarray(Image.open(pair_dir / "fixed_section53_v6_frame.png").convert("RGB"))
    moving = np.asarray(Image.open(pair_dir / "moving_section55_after_direct_stalign.png").convert("RGB"))
    if fixed.shape != moving.shape:
        raise ValueError(f"fixed/moving shape mismatch: {fixed.shape} vs {moving.shape}")
    fixed_od = _optical_density(fixed)
    moving_od = _optical_density(moving)
    overlay = np.zeros((*fixed_od.shape, 3), dtype=np.float32)
    overlay[..., 0] = fixed_od
    overlay[..., 1] = moving_od
    # Preserve the requested red/green semantics, with yellow where both H&E
    # images contain tissue signal.
    path = pair_dir / "overlay_after_he_red53_green55.png"
    overlay_image = Image.fromarray(np.clip(np.rint(overlay * 255.0), 0, 255).astype(np.uint8))
    overlay_image.save(path)
    visible = np.maximum(fixed_od, moving_od) > 0.04
    ys, xs = np.where(visible)
    if len(xs):
        pad = 8
        left = max(0, int(xs.min()) - pad)
        top = max(0, int(ys.min()) - pad)
        right = min(overlay_image.width, int(xs.max()) + pad + 1)
        bottom = min(overlay_image.height, int(ys.max()) + pad + 1)
        cropped = overlay_image.crop((left, top, right, bottom))
        cropped = cropped.resize((cropped.width * 3, cropped.height * 3), Image.Resampling.NEAREST)
        cropped.save(pair_dir / "overlay_after_he_red53_green55_cropped.png")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
