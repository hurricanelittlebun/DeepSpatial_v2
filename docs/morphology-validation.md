# Morphology implementation 验证记录

日期：2026-09-10。基线：`fdb9865`。修改位于当前工作树，未提交/推送。

## 历史实现验证（2026-09-10）

```bash
cd /data/buyonggan/DeepSpatial
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q --junitxml=artifacts/morphology-tests.xml
.venv/bin/python -m compileall -q deepspatial examples
git diff --check
```

结果：**26 passed, 4 warnings in 23.80s**。compileall 与 diff whitespace 检查均退出 0。

两项 CLI 的 `--help` 已执行成功。合成图生成命令 `OMP_NUM_THREADS=1 .venv/bin/python -m examples.morphology_path_demo` 已执行并查看输出：`artifacts/synthetic_morphology_path.png`。

## 覆盖与边界

| 用户要求 | 实际证据 |
|---|---|
| Original mode | 通过 git show 加载原版 GiT/Module，同 seed 初始参数、FM loss、ODE 输出逐元素一致；默认 UOT cost 与 hand-computed 值一致 |
| UNI2 feature query | 真实 HDF5、物理 XY 双线性和 Z 线性插值、lazy gather、越界处理 |
| Morphology-aware UOT | 实际 POT Sinkhorn 中交换 HE 特征，coupling 主方向随之交换 |
| Synthetic path | 中间层在 Z=3/10 处偏离直线，沿人工 morphology corridor |
| Endpoint | 缓存路径起终点恢复给定物理 XY |
| Velocity | PCHIP dx/dt 与有限差分一致；非均匀物理 Z，避免 dx/dz 混淆 |
| HE isolation | 非零空间 readout 下改变 h，vx 改变，vg/vc 完全不变且对 h 无直接梯度 |
| Cell type | 当时的四种模式分别测试 True/False；True 的 c head 有梯度，False 的 Lc/c 输入为零 |
| Toy E2E | 当时的 AnnData→dataset→POT→cached path→FM→GiT→optimizer→EMA→ODE→AnnData；四种模式×有无 cell type |
| Checkpoint | 手动 state dict 和真实 Lightning 一轮训练 checkpoint 均重载成功；fresh H5AD 无内部 normalized 字段仍可重建 |
| Reverse/adaptive ODE | 已知时间依赖速度的正向/反向位移正确；Dopri5 不跨越最终 H&E 边界 |
| Safeguards | UOT 在表达 densification 前拦截超大矩阵；基因顺序校验、路径复用/revision 失效、未完成 feature 临时组不被当作有效 section |
| Preprocessing | FOV 与各向异性 MPP crop、白色 padding、分批写出/query；冻结 encoder 行为 |

UNI2 wrapper 的单测只替换了受限的大模型权重构建，用真实 preprocessing/freezing/encode 接口验证；**未加载真实 681M 参数 UNI2 checkpoint**。其余 toy pipeline 使用真实代码及 POT/GiT/ODE，不是 mock pipeline。

## 4 条 warning

全部来自有意运行的 CPU Lightning smoke test：机器有 GPU 但选择 CPU、checkpoint 目录已含 config、小数据使用 num_workers=0、EMA 子模块在 eval 模式。没有数值失败或跳过测试。没有将这些 warning 静默过滤。

## 环境

仓库 `.venv` 通过 system-site-packages 复用 DriftST 的已有大型依赖；新增依赖及 NumPy 2.0.2 安装在 `.venv`，不修改原配准环境。

- Python 3.10.20
- torch 2.5.1+cu121（本次测试在 CPU 上）
- timm 1.0.25
- pytorch-lightning 2.6.5
- POT 0.9.7.post1
- torchdiffeq 0.2.5
- h5py 3.16.0
- anndata 0.11.4
- numpy 2.0.2（符合原 pyproject 的 `<2.1`）
- scipy 1.15.3
- pytest 9.1.1

继承环境里的 STalign 含一套严格旧版 dependency pins，pip 报告那些包的既有不兼容；本任务不导入或运行 STalign。不能据此声称整个共享环境通过 pip check。上述 DeepSpatial 测试均在最终 NumPy 2.0.2 下重新执行。

## 审查情况和未完成的真实实验

尝试独立只读 reviewer，但该工具因额度限制未返回结论；因此不声称经过独立审查。主执行者检查并补测了 fresh-H5AD/checkpoint 坐标恢复、基因顺序和 densification 前内存保护。

未运行：真实 UNI2 权重推理、真实 WSI feature extraction、全量组织训练、百万细胞 coupling、GPU 吞吐基准、生物学准确性评估。Dense UOT 仍然不适合百万细胞全对全计算；新增 sparse top-k 后端已通过 toy/API 回归，但还没有完成真实百万细胞吞吐和生物学准确性评估。见 `morphology.md` 的输入准备与限制。

## 发布清理后的重新验收（2026-10-06）

本次发布版本移除了 H&E late-fusion spatial-head API；H&E 仍可用于
morphology-aware UOT 和 morphology-guided path，原有 spatial/gene/cell-type
heads 以及 `use_celltype` 保留。当前公开入口为三种模式：`baseline`、`uot`
和 `path`（其中 `path` 同时启用 morphology UOT 与 morphology path）。

新鲜工作树上的验收结果：

```text
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q
164 passed, 6 warnings in 69.73s
PYTHONPATH=. .venv/bin/python -m compileall -q deepspatial examples scripts
exit 0
git diff --check
exit 0
```

`python -m build` 因当前 `.venv` 未安装 `build` 模块而无法执行；使用等价的
`pip wheel . --no-deps --no-build-isolation` 构建成功。wheel 包含
`deepspatial/histology`，不包含本地 `data/`、`weights/`、`artifacts/`、
`tests/` 或 `scripts/`。真实 UNI2 权重推理、真实 WSI 提取和全量组织训练仍
未在本次发布验收中执行。
