<p align="center">
  <img src="assets/logo.png" width="200" alt="DeepSpatial">
</p>

<p align="center">
  <a href="https://pypi.org/project/deepspatial/"><img src="https://img.shields.io/badge/Pypi-1.0.0-317EC2.svg" alt="PyPI"></a>
  <a href="https://yyh030806.github.io/DeepSpatial/"><img src="https://img.shields.io/badge/Homepage-deepspatial-f773a8.svg" alt="Homepage"></a>
  <a href="https://doi.org/10.64898/2026.04.28.721395"><img src="https://img.shields.io/badge/Paper-bioRxiv-00AAB5.svg" alt="BioRxiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-FF9800.svg" alt="License"></a>
</p>


**DeepSpatial** is a package for true 3D reconstruction of spatial omics tissues from serial 2D slices, built with `PyTorch` and designed to work smoothly with `AnnData`/`Scanpy` workflows.

## 3D spatial omics reconstruction

DeepSpatial provides an end-to-end framework for learning continuous 3D tissue representations:

- Reconstructs missing biological structure between adjacent sections.
- Jointly models spatial coordinates, gene expression, and cell identities.
- Supports large-scale training and sampling with GPU acceleration.
- Produces outputs that can be directly used in downstream single-cell/spatial analysis.

The package exposes a high-level API (`DeepSpatial`) for data setup, model training, and 3D reconstruction with minimal boilerplate.

## Installation

Recommended (PyPI):

```bash
pip install deepspatial
```

From source (development):

```bash
git clone https://github.com/hurricanelittlebun/DeepSpatial_v2.git
cd DeepSpatial_v2
pip install -e .
```

If you use GPU, install a PyTorch build matching your CUDA version.

## Quick start

1. Download an example dataset first: [Google Drive dataset folder](https://drive.google.com/drive/folders/11MICO4KmGRDlKfFf6LQ_aZC3mtF9NrNG?usp=sharing)

   Then place files under `data/merfish_mouse_hypothalamus/`.

2. Run DeepSpatial:

```python
import glob
import scanpy as sc
import deepspatial as ds

adatas = [
    sc.read_h5ad(p)
    for p in sorted(glob.glob("data/merfish_mouse_hypothalamus/merfish_*.h5ad"))
]

model = ds.DeepSpatial()
model.setup_data(adatas)
model.build_model()
model.fit(max_epochs=100)

adata_3d = model.reconstruct_full_volume(adatas, thickness=10.0)
```

## Resources

- Homepage: https://yyh030806.github.io/DeepSpatial/
- Bug reports and feature requests: https://github.com/hurricanelittlebun/DeepSpatial_v2/issues

## Citation

If you use DeepSpatial in your research, please cite:

```bibtex
@article {yang2026deepspatial,
	author = {Yang, Yuhang and Luo, Yiming and Zhang, Kai and Bu, Yonggan and Xia, Zheng and Peng, Haoxin and Yan, Rui and Liu, Qi and Chen, Yang and Shen, Lin and Chen, Enhong},
	title = {Reconstructing True 3D Spatial Omics at Single-Cell Resolution},
	year = {2026},
	doi = {10.64898/2026.04.28.721395},
	journal = {bioRxiv}
}
```

## Experimental morphology-aware extension

Frozen UNI2 preprocessing, morphology-aware UOT, and cached anatomical
correspondence paths are available as optional components. H&E constrains
endpoint correspondence and spatial paths; it is not a direct gene-expression
regressor or a separate spatial velocity head. The original cell-type
architecture and histology-off mode remain.

### What is improved in v2

Compared with the original DeepSpatial implementation, this version adds:

- Frozen UNI2-H morphology embeddings extracted offline from registered H&E
  patches in a common physical coordinate system.
- A morphology-aware UOT cost that combines spatial, gene, cell-type, and H&E
  morphology distances.
- A cached morphology-guided spatial path through intermediate H&E sections,
  using local candidate search and layered dynamic programming rather than a
  straight-line spatial interpolation.
- A sparse top-k UOT backend for large anchor slices, while retaining dense UOT
  as the small-scale reference implementation.
- Optional nucleus-supported paths and preservation of the original cell-type
  branch, including `use_celltype=True/False` ablation support.

H&E is deliberately not passed directly into the gene or cell-type velocity
heads. This keeps the extension focused on morphology-guided spatial transport
instead of turning it into an H&E-to-gene-expression regression model.

### How to call the v2 pipeline

Each independently registered tissue/g series is passed as an ordered list of
AnnData anchors. The anchors must share the same `.var_names`, use physical
micrometre coordinates in `.obsm['spatial_registered']`, and provide one
physical Z value in `.obs['z_um']` per section.

For the original DeepSpatial baseline:

```python
from deepspatial import DeepSpatial

model = DeepSpatial()
model.setup_data(anchors, spatial_key="spatial_registered", z_key="z_um",
                 label_key="cell_class", use_celltype=True)
model.build_model()
model.fit(max_epochs=100, save_dir="artifacts/model")
volume = model.reconstruct_full_volume(anchors, thickness=10.0)
volume.write_h5ad("artifacts/virtual_st.h5ad")
```

For morphology-aware UOT/path, first extract the frozen UNI2 feature store
from an already registered H&E manifest:

```bash
python -m deepspatial.histology.extract manifest.json artifacts/uni2.h5 \
  --checkpoint /path/to/UNI2/pytorch_model.bin \
  --device cuda:0 --batch-size 32
```

Then use the unified example entry point:

```bash
# Original baseline, no H&E
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad \
  --mode baseline

# H&E morphology-aware UOT
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad \
  --mode uot --features artifacts/uni2.h5 \
  --frame 00029_g0_registered

# H&E morphology-guided UOT + cached spatial path
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad \
  --mode path --features artifacts/uni2.h5 \
  --paths artifacts/paths.h5 --frame 00029_g0_registered \
  --uot-solver sparse_topk --uot-top-k 64
```

Use `--no-celltype` only for an explicit no-celltype ablation. The morphology
path mode requires a persistent path cache; the training loop does not read WSI
files or run UNI2 repeatedly.

- [Training entry point](examples/morphology_train.py)

This is a validated prototype, not a real-tissue validation result. No registration
is performed. For large anchor slices, use `setup_data(uot_solver="sparse_topk",
uot_top_k=64)`; the original dense UOT remains available as a small-scale
reference and still requires explicit size control.

## License

DeepSpatial is released under the MIT License.
