"""Synthetic QC only: inspect anatomical path bending and analytic velocity."""

from pathlib import Path
import tempfile
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deepspatial.histology import FeatureStore, MorphologyPathCache


def main():
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        store = FeatureStore(Path(directory) / "features.h5", mode="a")
        zs = [0.0, 3.0, 10.0]
        ys = [2.0, 4.0, 2.0]
        for sid, z, y in zip(["a", "m", "b"], zs, ys):
            f = np.zeros((7, 7, 2), np.float32)
            f[..., 1] = 1
            f[int(y), :, 0] = 1
            f[int(y), :, 1] = 0
            store.add_section(
                sid,
                f,
                z_um=z,
                origin_um=(0, 0),
                spacing_um=(1, 1),
                patch_size_um=1,
                mpp=(1, 1),
                coordinate_frame="synthetic",
            )
        cache = MorphologyPathCache(
            Path(directory) / "paths.h5",
            store,
            candidate_radius_um=2,
            candidate_spacing_um=1,
            move_weight=0.01,
            morphology_weight=10,
        )
        ids = cache.prepare(
            "a", "b", np.array([[1.0, 2.0]]), np.array([[5.0, 2.0]]), ["demo"]
        )
        t = torch.linspace(0, 1, 201, dtype=torch.float64)
        x, v = cache.evaluate(ids * len(t), t)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(t * 10, x[:, 1], color="crimson", label="Morphology PCHIP path")
        axes[0].plot([0, 10], [2, 2], "--", color="navy", label="Linear prior")
        axes[0].scatter(zs, ys, color="crimson", label="Matched morphology corridor")
        axes[0].set(
            xlabel="Physical Z (synthetic um)",
            ylabel="Y (synthetic um)",
            title="Fixed endpoints, curved intermediate path",
        )
        axes[0].legend(fontsize=8)
        axes[1].plot(t, v[:, 1], label="Analytic dy/dt")
        numerical = np.gradient(x[:, 1].numpy(), t.numpy())
        axes[1].plot(t, numerical, "--", label="Finite difference", alpha=0.8)
        axes[1].set(
            xlabel="Local t", ylabel="um / t", title="Velocity uses t, not physical Z"
        )
        axes[1].legend()
        fig.suptitle("SYNTHETIC validation — not real UNI2/tissue results")
        fig.tight_layout()
        fig.savefig(output / "synthetic_morphology_path.png", dpi=160)
        plt.close(fig)
    print(output / "synthetic_morphology_path.png")


if __name__ == "__main__":
    main()
