"""Train one independent tissue/g series. See docs/morphology.md for schema."""

import argparse
import anndata as ad
import numpy as np
import torch
from deepspatial import DeepSpatial
from deepspatial.histology import HistologyConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("anchors", nargs="+", help="H5ADs in increasing physical-Z order")
    p.add_argument("--mode", choices=["baseline", "uot", "path"], default="baseline")
    p.add_argument("--features")
    p.add_argument("--paths")
    p.add_argument("--frame")
    p.add_argument("--spatial-key", default="spatial_registered")
    p.add_argument("--z-key", default="z_um")
    p.add_argument("--section-key", default="section_id")
    p.add_argument("--label-key", default="cell_class")
    p.add_argument("--no-celltype", action="store_true")
    p.add_argument("--pairs", type=int, default=50000)
    p.add_argument("--uot-solver", choices=["dense", "sparse_topk"], default="dense")
    p.add_argument("--uot-top-k", type=int, default=None)
    p.add_argument("--uot-candidate-radius", type=float, default=None)
    p.add_argument("--no-uot-bidirectional", action="store_true")
    p.add_argument(
        "--allow-linear-path-fallback",
        action="store_true",
        help="Use an auditable spatial-only path outside the morphology feature grid",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="artifacts/model")
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = HistologyConfig(
        use_histology=args.mode != "baseline",
        use_morphology_uot=args.mode != "baseline",
        use_morphology_path=args.mode == "path",
        feature_path=args.features,
        path_cache=args.paths,
        coordinate_frame=args.frame,
        section_key=args.section_key,
        allow_linear_fallback=args.allow_linear_path_fallback,
        seed=args.seed,
    )
    model = DeepSpatial()
    model.setup_data(
        [ad.read_h5ad(path) for path in args.anchors],
        spatial_key=args.spatial_key,
        z_key=args.z_key,
        label_key=args.label_key,
        n_samples_base=args.pairs,
        uot_solver=args.uot_solver,
        uot_top_k=args.uot_top_k,
        uot_bidirectional=not args.no_uot_bidirectional,
        uot_candidate_radius=args.uot_candidate_radius,
        use_celltype=not args.no_celltype,
        histology=config,
    )
    model.build_model()
    model.fit(max_epochs=args.epochs, save_dir=args.save_dir, save_ckpt=True)


if __name__ == "__main__":
    main()
