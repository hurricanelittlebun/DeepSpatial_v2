"""Audit existing 00029/g0 registration transforms without re-registering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from deepspatial.histology.registered_coordinates import (
    crop_px_to_source_um,
    registered_to_source_points,
    source_to_registered_points,
    source_um_to_crop_px,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/00029_g0")
    parser.add_argument(
        "--registration-root",
        required=True,
        help="external DeepSpatial_v2 registration directory",
    )
    parser.add_argument("--through-section", type=int, default=11)
    parser.add_argument("--sample-size", type=int, default=4000)
    args = parser.parse_args()

    # This is the only project-specific dependency: it restores the saved
    # STalign maps in exactly the same way as the registration pipeline.
    from deepspatial_v2.data_export.cell_level_alignment import (
        SavedTransformStore,
        load_section_transform,
    )

    dataset = Path(args.dataset).resolve()
    output = dataset / "qc"
    output.mkdir(parents=True, exist_ok=True)
    sections = pd.read_parquet(dataset / "he_sections.parquet")
    sections = sections[sections["section_id"] <= args.through_section]
    store = SavedTransformStore(args.registration_root, device="cpu")
    rng = np.random.default_rng(0)
    records: list[dict] = []
    registered_samples: dict[int, np.ndarray] = {}

    for row in sections.itertuples(index=False):
        mask = np.asarray(Image.open(row.source_mask_path).convert("L")) > 0
        image_size = Image.open(row.source_image_path).size
        if image_size != (mask.shape[1], mask.shape[0]):
            raise ValueError(
                f"section {row.section_id}: image/mask dimensions differ: "
                f"{image_size} versus {(mask.shape[1], mask.shape[0])}"
            )
        yy, xx = np.nonzero(mask)
        count = min(int(args.sample_size), len(xx))
        chosen = rng.choice(len(xx), size=count, replace=False)
        crop_px = np.column_stack([xx[chosen], yy[chosen]]).astype(float)
        mpp = (row.analysis_pixel_size_x_um, row.analysis_pixel_size_y_um)
        bbox = (row.bbox_x0, row.bbox_y0)
        source_um = crop_px_to_source_um(crop_px, mpp=mpp, bbox_origin_px=bbox)
        section = load_section_transform(row.section_transform_path)
        registered = source_to_registered_points(source_um, section, store)
        recovered = registered_to_source_points(registered, section, store)
        error_um = np.linalg.norm(recovered - source_um, axis=1)
        recovered_px = source_um_to_crop_px(recovered, mpp=mpp, bbox_origin_px=bbox)
        error_px = np.linalg.norm(recovered_px - crop_px, axis=1)
        registered_samples[int(row.section_id)] = registered
        records.append(
            {
                "section_id": int(row.section_id),
                "z_um": float(row.z_um),
                "n_edges": len(section.edge_chain),
                "forced_flip": bool(section.forced_flip),
                "roundtrip_median_um": float(np.median(error_um)),
                "roundtrip_p95_um": float(np.percentile(error_um, 95)),
                "roundtrip_max_um": float(np.max(error_um)),
                "roundtrip_p95_source_px": float(np.percentile(error_px, 95)),
            }
        )

    report = pd.DataFrame(records)
    report.to_csv(output / "registration_coordinate_audit.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(registered_samples)))
    for color, (section_id, points) in zip(colors, registered_samples.items()):
        axes[0].scatter(
            points[:, 0], points[:, 1], s=0.45, alpha=0.22, color=color,
            label=str(section_id), rasterized=True,
        )
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    axes[0].set_title("Existing registered tissue samples (sections 1–11)")
    axes[0].set_xlabel("registered x (µm)")
    axes[0].set_ylabel("registered y (µm)")
    axes[0].legend(title="section", markerscale=5, ncol=2)
    axes[1].plot(report.section_id, report.roundtrip_p95_um, "o-", label="p95")
    axes[1].plot(report.section_id, report.roundtrip_max_um, "o--", label="max")
    axes[1].set_title("Forward → inverse coordinate error")
    axes[1].set_xlabel("section")
    axes[1].set_ylabel("error (µm)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output / "registration_coordinate_audit.png", dpi=220)
    plt.close(figure)

    summary = {
        "sections": report.section_id.tolist(),
        "max_p95_roundtrip_um": float(report.roundtrip_p95_um.max()),
        "max_roundtrip_um": float(report.roundtrip_max_um.max()),
        "note": "Audit applies saved transforms only; it does not estimate registration.",
    }
    (output / "registration_coordinate_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
