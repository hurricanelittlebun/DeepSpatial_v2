# DeepSpatial：UNI2 morphology-aware UOT 与 spatial flow

这版修改不重新配准，也不是 H&E→gene-expression 回归模型。H&E 约束端点配对与空间运动；基因和 cell-type 的训练目标仍由 ST 两端决定。

## 1. 原仓库实际实现

基于 release commit `fdb9865`：

- `core.setup_data` 在全部 anchors 上分别归一化 X、Y、Z；模型输入不是微米。各轴分别缩放，不能把模型坐标直接用于 WSI 查询。
- `uot_solver` 使用 POT unbalanced Sinkhorn：均匀边缘，`reg=max(uot_reg,.01)`、`reg_m=uot_tau`、100 次迭代、阈值 `1e-3`。原 cost 是 `alpha_spatial * normalized_euclidean + (1-alpha_spatial) * normalized_gene_cosine + 10 * class_mismatch`；空间/基因距离各自除以最大值加 epsilon。
- `dataset` 对相邻 ST anchors 建立稠密 coupling，按 coupling 概率有放回抽样；`n_samples_base` 是全部间隔的目标总量，按相邻细胞数乘积分配并取整，不是每个间隔各抽这么多。数据集存储展开后的 x/g/c/z 两端张量。
- `module._shared_step` 是真实训练入口；并非通用 `transport.training_losses`。Linear 模式下 x/g/c/z 都线性插值，target 是两端差值；随机 t 在 `[0,1]`。
- GiT 把 XY token 与 gene tokens 放入共享 Transformer；t、z、delta_z 和连续 one-hot c 做 conditioning。空间/基因使用 FinalLayer，cell-type 使用共享 token 均值的线性 head。
- loss 是 `Lx + 0.1 Lg + 10 Lc`，权重原本就可配置。EMA 网络负责 ODE sampling，联合积分 `[x,g,c]`。
- reconstruction 从两端随机选取起始细胞、积分并插值，最后保留亲本对应数量的非零基因。`thickness` 控制生成细胞密度，**并非保证输出固定等距 Z 平面**；这项原有语义未改。

原代码缺标签时没有拟合 LabelEncoder，`setup_data` 读取 `classes_` 会失败；重建也强制读取标签。现在默认要求完整标签并明确提示 `use_celltype=False`；关闭时保留一维 unknown 占位状态、全部 cell-type 模块，但输入 c=0、Lc=0、积分 vc=0，输出标签 unknown。

## 2. 输入契约：每次一条独立 tissue/g 序列

每个 anchor 是一个 AnnData：

- `.X`：经过用户选定预处理的 ST 表达矩阵；这里不自动改变 count/log/normalization 定义。
- `.var_names`：所有 anchors 必须相同且顺序相同。
- `.obsm[spatial_key]`：同一配准坐标系的浮点 XY，单位 **µm**。
- `.obs[z_key]`：真实物理 Z，单位 **µm**；单个 anchor 唯一，列表严格递增。
- `.obs[section_key]`：单个 anchor 唯一 section ID，与 feature store 完全一致。
- `.obs[label_key]`：`use_celltype=True` 时完整的细胞类型标签。

不同 g 具有独立 frame 时，分别训练/建 store。不要把它们的局部 XY 当成一个已拼合的器官。不能把 `z_index` 静默当作 `z_um`，需要实验切片间距或显式物理 Z 表。

## 3. UNI2 离线提取

`UNI2Encoder` 明确加载 **UNI2-h**：ViT-H/14、1536 维 CLS embedding。结构和图像预处理依据 [UNI2 官方模型卡](https://huggingface.co/MahmoodLab/UNI2-h)。支持获批的本地 `pytorch_model.bin` 或用户已获授权的 Hugging Face 下载。始终冻结参数、eval、inference_mode；不属于 DeepSpatial optimizer。

```python
from deepspatial.histology.uni2 import UNI2Encoder
encoder = UNI2Encoder('/path/to/approved/pytorch_model.bin', device='cuda:0')
h = encoder.encode(pil_rgb_patches, batch_size=32)  # CPU [N,1536]
```

模型权重需要用户自己的授权，不随此仓库分发。UNI2 的使用限制与本仓库 MIT 许可证不同，请遵守模型卡的授权条款。

提取路线：already-registered WSI → 物理坐标网格 → 固定物理 FOV → UNI2 → HDF5。`patch_size_um / mpp_xy` 决定 crop 像素宽高；透明 padding 合成白色。各向异性 MPP 的物理正方形重采样为 224×224。所有 section 的物理 FOV 必须相同。

命令行 manifest 示例（以下数值仅示范格式，不代表你的实际标定）：

```json
[
  {
    "registered": true,
    "image_path": "/absolute/registered_section.ome.tif",
    "section_id": "00029_g0_11",
    "z_um": 50.0,
    "mpp": [0.5, 0.5],
    "image_origin_um": [0.0, 0.0],
    "grid_origin_um": [56.0, 56.0],
    "grid_shape": [200, 160],
    "grid_spacing_um": [25.0, 25.0],
    "patch_size_um": 112.0,
    "coordinate_frame": "00029_g0_registered",
    "valid_mask_path": "/absolute/grid_tissue_mask.npy"
  }
]
```

`grid_shape` 是 `[H_y,W_x]`；spacing/origin/mpp 是 `[x,y]`；grid origin 表示第一个 **patch 中心**，image origin 表示 registered raster 的像素坐标原点。mask 是与 feature grid 一样大的 bool 数组。全 FOV 越过图像边缘的点会排除。未提供 mask 时不会自动做组织检测，背景需要上游 QC 排除。

```bash
python -m deepspatial.histology.extract manifest.json artifacts/uni2.h5 \
  --checkpoint /path/to/approved/pytorch_model.bin --device cuda:0 --batch-size 32
```

写入 `uni2.h5.log`；完整 section 可复用，输入路径/大小/mtime/manifest 变化时拒绝复用，要求新 store。该元数据检查不是文件内容 SHA 校验。HF 自动下载模式不固定权重 revision；严格复现建议使用固定本地 checkpoint。

CLI 使用 OpenSlide，系统须安装 OpenSlide native library。**不能把 raw SDPC 冒充 registered WSI**。如果目前只有原图和已保存 affine/STalign transforms，可向 `precompute_section` 传入应用现有变换的 registered reader（`read_region`、`dimensions` 契约），或先导出 registered raster；这里不新增配准算法，也没有假装你的全分辨率 registered WSI 已经导出。

## 4. Feature store

HDF5：`sections/<hashed_section_id>/features [H,W,D]` float32 LZF chunked，`valid [H,W]`，元数据含 ID、Z、grid origin/spacing、FOV、MPP、frame 和 provenance。ID 用 hash 作路径，不受 `/` 等字符影响。一个 store 只允许一个 frame、一个 section/physical Z。

```python
from deepspatial.histology import FeatureStore
store = FeatureStore('artifacts/uni2.h5')
h = store.get_feature('00029_g0_11', xy_um)  # [N,2] -> [N,D]
h = store.query(xy_um, z_um)                # 标量 Z 或每行一个 Z
h, valid = store.query(xy_um, z_um, return_valid=True)
```

XY 双线性、Z 相邻平面线性插值。非法/背景/范围外查询默认抛错；可以显式取 valid mask。不会将越界点悄悄贴到边缘组织。小平面使用有上限 CPU LRU，过大平面按所需行/列读取；没有长期打开的 HDF5 文件句柄。返回 tensor 与输入的 device/dtype 一致，但此版 feature gathering 在 CPU 上，GPU 输入仍有同步开销。

## 5. Morphology-aware UOT

`Dh = clip(1 - normalize(h0) @ normalize(h1).T, 0, 2)`。

`C = ws * Ds_norm + wg * Dg_norm + wh * Dh_norm + wc * Dclass`。

为保持 release 行为，各连续距离采用原版 max normalization，`D / (max(D)+1e-9)`，不改为新的 P95 normalization。保留 class mismatch 的默认 10 倍权重。零 morphology 向量视为无效；空 gene 向量的 undefined cosine 设为 1，避免 NaN 污染 coupling。

这里有一个原实现与你的公式之间的尺度区别：原 Ds 在 registered XY **按轴归一化后**计算，而不是原始 µm。默认 `spatial_metric='legacy_normalized'` 沿用原 metric，因此仅改变 `histology_weight` 不会偷偷改变 Ds，`histology_weight=0` 与原 cost 退化一致。若要求物理各向同性距离，显式设 `spatial_metric='physical_um'`；进行对照实验时各组保持同一个 metric（包括 wh=0 的对照）。所有 H&E 查询和 DP 搜索始终使用真实 µm，不受此 cost 选项影响。

当前同时保留两种 UOT 后端：`uot_solver='dense'` 使用原始 POT 全对全 coupling，`uot_solver='sparse_topk'` 使用注册 XY 的空间候选图和稀疏 Sinkhorn。`max_uot_entries` 只限制 Dense 后端；减少 `n_samples_base` **不会**减少 Dense UOT 矩阵大小。

### 5.1 Spatial top-k sparse UOT

稀疏后端不把不存在的配对填成零 cost，而是只保留候选边：

1. 用 `cKDTree` 为每个 source cell 找 target 中最近的 `uot_top_k` 个细胞；
2. 默认再加入反向 kNN，保证两侧细胞都有候选支持；
3. 只在这些边上计算空间、gene、class 和 UNI2 morphology cost；
4. 使用稀疏 KL-relaxed Sinkhorn，返回 edge-list coupling；
5. dataset 直接按 edge mass 抽样，不执行 `pi.ravel()`。

推荐先比较 `uot_top_k=32/64/128`。初始运行示例：

```python
ds.setup_data(
    ordered_anchor_adatas,
    spatial_key='spatial_registered',
    z_key='z_um',
    label_key='cell_class',
    uot_solver='sparse_topk',
    uot_top_k=64,
    uot_bidirectional=True,
    histology=cfg,
)
```

`uot_candidate_radius` 是可选的候选半径：有 histology runtime 时，候选图始终使用 registered physical XY，因此单位是 µm；没有 histology runtime 时，单位是 UOT solver 输入的 normalized XY。半径过小的细胞会保留最近邻 fallback，不会产生无支持的 marginal。

Dense 与 sparse 的 coupling 不要求数值完全相同；当 `uot_top_k` 覆盖全部 target cells 时，当前 sparse Sinkhorn 与 POT Dense reference 的数值回归一致。正式大规模运行仍应检查 candidate coverage、coupling mass、匹配距离和切片叠加 QC。

## 6. Morphology path 与缓存

对已抽取的 UOT pair，遍历两端之间 store 中的全部 H&E 层：

1. 按实际 `t_k=(z_k-z0)/(z1-z0)` 得到线性 coarse XY。
2. 中间层在 coarse XY 周围建立物理局部方形网格，删除 invalid candidates；端点固定。
3. 相邻层 edge 为 `move_weight * squared_distance_um + morphology_weight * cosine_distance`。
4. 分层动态规划与回溯得到离散 anatomical correspondence path。
5. 用 PCHIP 插值，以局部 **t** 为 knot 坐标，导数天然是 **dx/dt**。转换回模型坐标时逐轴除以 XY range；没有把 dx/dz 错当作 target。

起点/终点保证精确；即使端点细胞落在 H&E 有效 mask 外，也保留其 UOT 坐标，并在该段只使用 movement cost。中间层若无有效候选则明确失败，不静默回退为直线。不保证 **学出来的 ODE** 必然命中相同终点，也不保证 PCHIP 的每个中间点都落在 tissue mask 内。关闭 morphology path 时才走原始 Linear path。

缓存：`paths/<content_hash>/t [K]`、`xy_um [K,2]`、`coefficients [4,K-1,2]` float64；metadata 记录 endpoint IDs/coordinates、section IDs、feature revision、参数与格式版本。相同 pair 不重复求解；feature revision 或参数变化产生新 key。写入使用临时组，完成后才发布正式 key；不支持多进程同时写同一个 HDF5。磁盘/文件系统级损坏不在此事务机制保证之内。

dataset 存 `path_id [N]` int64，runtime 将其映射到持久 hash。训练批次按 knot 集合分组，torch 向量化求值多项式和导数；没有训练时 shortest path。断点恢复训练应先用相同输入/seed/feature store 重新 setup_data，然后使用缓存及 Lightning resume。

这不是 cell lineage，也不是跨切片跟踪同一个真实细胞。

## 7. Flow Matching / GiT / sampling

空间使用 cached `xt, ux`；基因、c、z 仍调用原 `path_sampler.plan`。H&E 特征只用于 morphology-aware UOT 和缓存的 anatomical path，不作为 GiT 的额外输入。

ODE 每次在当前积分位置查询缓存，无 WSI/UNI2/DP。新模式反向积分使用 `t_model=1-t`、正的 anchor gap、三个速度取负。原版反向调用的负 gap/不取负行为只在 baseline 保留以便原版回归。自适应 RK 强制在 t=1 落点，初始内部步长 .01；数值边界处 H&E Z conditioning 延拓到端点，输出仍在 `[0,1]`。

checkpoint 同时保存训练坐标统计和 gene names/order。重新加载原始 H5AD 做重建时，会用保存的统计生成内部 normalized 坐标，不会重新拟合缩放，也不要求 H5AD 已带 `spatial_norm/z_norm`。基因顺序不符则拒绝运行，防止表达输出对应错基因。

第一版 H&E 只支持 **Linear molecular/cell-type path + velocity prediction + ODE**；拒绝 VP/GVP 和 SDE 的线性 score 转换，因为不能将那套转换机械套到 DP path 上。普通 baseline transport 代码保持原有选项。

### 7.1 按 H&E 分割细胞输出预测

`predict_on_he_cells()` 是单细胞级推理入口，不使用 QC 图中的 `max_edges_plot`，也不把 `chunk_size` 当作输出数量限制。它从左侧 ST anchor 的全部细胞逐批运行 ODE，在每个 H&E Z 平面上建立传播后的位置索引，再将每个 H&E 查询细胞分配到最近的传播轨迹：

```python
virtual = model.predict_on_he_cells(
    he_cells,
    anchor_1,
    anchor_11,
    steps=40,
    chunk_size=256,
    device="cuda",
)
```

返回对象满足 `virtual.n_obs == he_cells.n_obs`，并保留 H&E 的 registered XY/Z。`obsm[spatial_key]` 是 H&E 查询坐标，`obsm["spatial_flow"]` 是被分配的传播轨迹坐标，`obs["prediction_distance_um"]` 用于 QC；`.X` 是预测表达矩阵。该接口表达的是空间/解剖对应，不是 cell lineage，也不是同一个真实细胞的跨切片追踪。多个 H&E 细胞可以暂时分配到同一条近邻传播轨迹，因此后续正式版本还应结合局部密度和双向 anchor 结果做去重/融合。

## 8. Tensor shapes

| 变量 | shape | 单位/语义 |
|---|---|---|
| x0/x1/xt/ux/vx | `[B,2]` | normalized XY，velocity 是 normalized XY / local t |
| g0/g1/gt/ug/vg | `[B,G]` | 原 ST 表达空间 |
| c0/c1/ct/uc/vc | `[B,C]` | 原 one-hot 连续状态；无标签 C=1 |
| z0/z1/zt/delta_z | `[B,1]` | normalized physical Z |
| t | `[B]` | anchor-local `[0,1]` |
| UNI2 ht | `[B,1536]` | 冻结特征；合成测试允许小 D |
| H | `[B,1+ceil(G/patch_size),hidden_size]` | 共享 backbone |
| feature grid | `[H_y,W_x,1536]` | µm 网格 |
| UOT cost/coupling | Dense `[N0,N1]`；sparse `row/col/mass [E]` | `E` 约为 `N*top_k` |
| reconstructed trajectory | `[steps,B,2/G]` | model space，导出时恢复物理坐标 |

## 9. 新配置与默认值

统一由 `HistologyConfig`（或同字段 dict）传入 `setup_data(histology=...)`。

| 字段 | 默认 | 用途 |
|---|---|---|
| use_histology | False | 主开关；关闭时禁止打开子开关 |
| use_morphology_uot | False | morphology cost |
| use_morphology_path | False | DP/PCHIP spatial target |
| feature_path / path_cache | None / None | HDF5 文件；启用相应模式时必需 |
| section_key | section_id | anchor ID 列 |
| coordinate_frame | None | 必须显式指定并与 store 一致 |
| spatial_weight / gene_weight | None / None | None 继承 alpha / 1-alpha |
| histology_weight / class_weight | 1 / 10 | cost 权重 |
| spatial_metric | legacy_normalized | 保留原 registered XY metric；可显式选择 physical_um |
| candidate_radius_um / candidate_spacing_um | 100 / 25 | 局部网格半径/间距 |
| move_weight / morphology_weight | .001 / 1 | DP edge；move_weight 数值以 µm² 为尺度 |
| max_candidates | 1024 | 每层局部网格上限 |
| max_uot_entries | 25000000 | 单个 coupling 矩阵条目上限 |
| feature_cache_mb | 512 | CPU feature LRU 上限 |
| seed | 0 | 新模式 pair sampling、torch/Lightning seed |

UOT 后端参数由 `setup_data` 传入：`uot_solver='dense'`（默认）、`uot_top_k=None`、`uot_bidirectional=True`、`uot_candidate_radius=None`。其中 `uot_top_k` 在 sparse 后端必须为正整数。

`setup_data(use_celltype=True)` 默认保留 cell-type 分支；False 是显式 ablation。原 alpha_spatial、uot_reg、uot_tau、lambda_g、lambda_c、网络维度等参数保持原 API。

预处理另有 `patch_size_um`、`mpp [x,y]`、image/grid origin、grid spacing/shape、valid_mask、batch_size=32、device、checkpoint。这些是数据/编码设置，不是训练中实时调参。

## 10. 三种运行方式

安装：`python -m pip install -e '.[histology,test]'`。本次独立环境是仓库 `.venv`，不改变已有配准环境；真实运行前需准备获批 UNI2 权重和符合上述单位/schema 的 store。

```python
from deepspatial import DeepSpatial
from deepspatial.histology import HistologyConfig

mode = 'path'  # baseline / uot / path
cfg = HistologyConfig(
    use_histology=mode != 'baseline',
    use_morphology_uot=mode != 'baseline',
    use_morphology_path=mode == 'path',
    feature_path='/absolute/uni2.h5',
    path_cache='/absolute/paths.h5',
    coordinate_frame='00029_g0_registered',
)
ds = DeepSpatial()
ds.setup_data(ordered_anchor_adatas, spatial_key='spatial_registered', z_key='z_um',
              label_key='cell_class', use_celltype=True, histology=cfg)
ds.build_model()
ds.fit(max_epochs=100, save_dir='artifacts/model', save_ckpt=True)
volume = ds.reconstruct_full_volume(ordered_anchor_adatas, thickness=5., steps=100)
volume.write_h5ad('artifacts/virtual_st.h5ad')
```

直接在仓库运行下面四条中的一条（替换实际文件路径；没有 cell type 时加 `--no-celltype`）：

```bash
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad --mode baseline
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad --mode uot --features uni2.h5 --frame 00029_g0_registered
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad --mode path --features uni2.h5 --paths paths.h5 --frame 00029_g0_registered
python -m examples.morphology_train anchor11.h5ad anchor21.h5ad --mode path --features uni2.h5 --paths paths.h5 --frame 00029_g0_registered --uot-solver sparse_topk --uot-top-k 64
```

baseline 不需要 H&E/权重。后三种运行时都只需要已经提取的 feature store，不需要加载 UNI2。

## 11. 文件改动

| 文件 | 内容 |
|---|---|
| core.py | histology/config 接线、物理单位检查、cell-type 开关、metadata/checkpoint、反向采样标识 |
| data_utils/dataset.py | 可选 HE cost、缓存 path IDs、显式缺标签处理 |
| data_utils/uot_solver.py | Dense release solver、空间 top-k 候选、稀疏 cost、稀疏 Sinkhorn、edge coupling |
| module.py | 空间 path target、反向 ODE，以及原有 g/c 分支 |
| models/git.py | 原有 spatial/gene/cell-type heads；不接收 H&E 特征 |
| transport/transport.py、integrators.py | 仅传递可选 ODE options，baseline 为 None |
| histology/config.py | 配置与边界检查 |
| histology/feature_store.py | chunked store、物理 XY/Z 查询、validity/LRU |
| histology/uni2.py | 冻结的官方 UNI2-h wrapper |
| histology/preprocessing.py、extract.py | 物理 FOV 提取和可重跑 CLI |
| histology/path.py | DP/PCHIP/内容寻址持久 cache |
| histology/runtime.py | µm↔model normalized bridge |
| examples/morphology_train.py | 三种模式的统一入口 |
| tests/ | 原版回归、特征/路径/信息隔离/cell-type/真实 API/ODE/checkpoint 测试 |

没有改变 registration，没有训练 H&E→gene 网络，没有删除 `transport/path.py` 的 ICPlan。

## 12. 验证、性能和未验证项

运行 `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q`。详见同目录 `morphology-validation.md` 的实际运行记录。

主要瓶颈/限制：

- Dense UOT 的 O(N0*N1) 时间/内存仍是最大限制，多个矩阵同时存在；默认 guard 的 2500 万条目也可能占用数 GB。Sparse top-k 将候选规模降到约 O(N*top_k)，但 top-k 过小会排除真实对应，必须结合 coverage 和叠加图检查。
- 当前 sparse Sinkhorn 使用 CPU `scipy.sparse` 矩阵乘；候选 edge list 仍需内存，gene 表达仍在 dataset 中转成 dense。它是可运行的大规模后端，不应直接宣称已经完成百万细胞 GPU 极限优化。
- 原 dataset 会将 sparse expression 转 dense，并按 sampled pair 复制端点表达；这是原有结构，未在本次大规模重构。
- 每条 unique pair 的 DP 为 O(K*M²)，局部网格与 cache 避免整图搜索及训练重复计算，但 5 万条长路径仍可能预处理较慢。
- Feature queries 支持 batch，但 HDF5/插值主要 CPU，GPU 训练会发生 device 同步；路径多项式求值为 torch，可在 GPU 上运行。
- 预处理每次只读 patch，但暂时在内存中保留一个 section 的完整 float32 feature grid；不是无限规模的 streaming writer。应按实际 grid 大小规划内存，后续可增加分块 writer。
- PCHIP 只保证分段光滑和端点，不是解剖正确性的证明；染色差异、空洞、错误配准、重复纹理可能诱导错误路径。mask 边界处的 ODE 中间状态可能出界而显式报错。
- 原 UOT class cost=10 可能压过 morphology；需要在独立验证集上选择权重，不能将合成测试当作组织效果验证。
- 严格信息隔离指 fixed-state 下没有直接 HE→vg/vc；通过 spatial state/UOT coupling 仍然存在预期的间接影响。
- 原版 sampling 的反向语义只在新模式修正，因此 morphology path 模式与 baseline 的重建差异不只来自 morphology；训练 loss 的关闭模式回归保持一致。
- 此次没有真实 UNI2 checkpoint 推理，没有真实 WSI 特征提取，没有全量真实数据训练或生物学效果评估。wrapper 测试明确使用微型替身替代受限权重构建，其他 toy pipeline 使用真实 dataset/POT/DP/GiT/ODE/AnnData。

实际数据接入前需核验：真实物理 Z、每组独立 frame、最终 H&E 注册 reader/raster、组织/QC mask、获批 UNI2 权重和表达预处理方案。
