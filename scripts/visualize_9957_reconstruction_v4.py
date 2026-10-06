"""Render the final 9957/g0 reconstruction with DeepSpatial visual utilities.

The H5AD can contain hundreds of thousands to millions of cells.  The
interactive figures therefore use a deterministic point sample while the
manifest records the full H5AD shape and the rendered sample size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import plotly.express as px

# Direct execution places ``scripts/`` rather than the repository root on
# sys.path.  Add the root explicitly so this utility works outside an
# editable install as well.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.vis_utils import (
    interactive_3d_labels,
    plot_3d_labels,
    plot_orthogonal_projections,
)


DEFAULT_H5AD = Path(
    "/data/buyonggan/DeepSpatial/data/9957_g0/"
    "reconstruction_final_thickness10_v4_manual_rotation/"
    "adata_9957_g0_full_thickness10_celltype_domain_niche.h5ad"
)
DEFAULT_OUTPUT = DEFAULT_H5AD.parent / "visualizations"


def _palette(categories: list[str]) -> dict[str, str]:
    colors = (
        list(px.colors.qualitative.Dark24)
        + list(px.colors.qualitative.Light24)
        + list(px.colors.qualitative.Set3)
    )
    return {category: colors[i % len(colors)] for i, category in enumerate(categories)}


def _render_interactive(adata, column: str, output: Path, max_points: int) -> None:
    categories = sorted(adata.obs[column].astype(str).unique().tolist())
    figure = interactive_3d_labels(
        adata,
        color_col=column,
        palette=_palette(categories),
        spatial_key="spatial_st_corrected",
        z_key="z_um",
        point_size=1.4,
        opacity=0.72,
        bg_color="white",
        max_points=max_points,
        title=f"9957/g0 reconstructed 3D — {column} (sampled)",
        width=1500,
        height=950,
        save_html=None,
    )
    # Keep each HTML self-contained so it can be opened on a disconnected
    # server or copied to another machine.
    figure.write_html(output, include_plotlyjs=True, full_html=True)


def _render_static_domain(adata, output: Path, max_points: int) -> None:
    categories = sorted(adata.obs["pred_domain"].astype(str).unique().tolist())
    palette = _palette(categories)

    figure = plot_3d_labels(
        adata,
        color_col="pred_domain",
        palette=palette,
        spatial_key="spatial_st_corrected",
        z_key="z_um",
        elev=25,
        azim=-58,
        z_stretch=1.0,
        point_size=0.35,
        alpha=0.55,
        max_points=max_points,
        bg_color="white",
        show=False,
    )
    figure.savefig(output / "domain_3d_static.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    plot_orthogonal_projections(
        adata,
        color_col="pred_domain",
        palette=palette,
        spatial_key="spatial_st_corrected",
        z_key="z_um",
        point_size=0.28,
        alpha=0.55,
        max_points=max_points,
        bg_color="white",
        save_png=str(output / "domain_orthogonal_projections.png"),
        show=False,
    )
    plt.close("all")


def render(h5ad_path: Path, output: Path, max_points: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    required = {
        "spatial_st_corrected",
        "z_um",
        "cell_class",
        "pred_domain",
        "pred_niche",
    }
    missing = sorted(
        key for key in required if key not in adata.obsm and key not in adata.obs
    )
    if missing:
        adata.file.close()
        raise ValueError(f"Missing visualization fields: {missing}")

    outputs = {
        "cell_class": output / "cell_class_3d_interactive.html",
        "pred_domain": output / "domain_3d_interactive.html",
        "pred_niche": output / "niche_3d_interactive.html",
        "domain_static_3d": output / "domain_3d_static.png",
        "domain_orthogonal": output / "domain_orthogonal_projections.png",
    }
    for column, path in (
        ("cell_class", outputs["cell_class"]),
        ("pred_domain", outputs["pred_domain"]),
        ("pred_niche", outputs["pred_niche"]),
    ):
        _render_interactive(adata, column, path, max_points)
    _render_static_domain(adata, outputs["domain_static_3d"].parent, max_points)

    manifest = {
        "source_h5ad": str(h5ad_path.resolve()),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "spatial_key": "spatial_st_corrected",
        "z_key": "z_um",
        "render_max_points": int(max_points),
        "sampling": "deterministic random sample from DeepSpatial vis_utils",
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    adata.file.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-points", type=int, default=150000)
    args = parser.parse_args()
    manifest = render(args.h5ad.resolve(), args.output.resolve(), args.max_points)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
