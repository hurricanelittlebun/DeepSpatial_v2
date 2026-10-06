# 9957 坐标系修复、QC 与重建计划

## Goal

修复 9957/g0 在 section 51→61 处出现的坐标系断裂，重新物化与统一坐标系一致的 ST 锚点、UNI2 morphology feature、核 feature、UOT cache 和 morphology/nucleus path cache。所有新结果使用独立的 `v5_coordinate_repair` 目录，不覆盖已有 v1/v4 结果。先生成对齐 QC 图，等待用户确认后再训练和完整重建。

## Spec / source of truth

- 用户当前请求与本轮对话中的 9957 坐标系诊断结论。
- 现有代码与数据：`deepspatial/`、`scripts/`、`examples/`、`data/9957_g0/`。
- 现有坏版本仅作输入和对照，不修改其内容。

## Architecture and constraints

- 目标坐标系采用用户已确认的手工 H&E v4 frame，并把同一个全局 affine 同步施加到原来仍在旧 frame 的 1–54 sections。
- 新坐标系命名为 `9957__g0_registered_manual_rotation_v4_global_repaired`，防止误把旧 v4 feature store 当成新 frame。
- ST 只做坐标字段修复；保留原始/候选坐标列和完整 provenance。
- H&E/UNI2/核 feature store 通过现有 `frame_resample` 工具进入统一 v4 frame：对 v4 provenance 标记为 unchanged/pre-manual 的 section 使用手工 affine，对已经位于 v4 frame 的 section 使用 identity。
- 重建前必须重新生成 UOT 和 path cache，使训练不再读取旧 v4 cache。
- 不启动训练/重建，直到 QC 图生成并由用户确认。
- 不删除、不覆盖任何既有结果。

## Review focus

- 51→61 是否回到连续统一坐标系；
- ST、H&E feature、核 feature 是否使用同一 frame metadata；
- 逆 affine 是否只应用于曾被 v4 手工 affine 影响的 section；
- 新 UOT/path cache 是否引用新 anchor/store/frame；
- QC 是否能显示修复前后差异及 section 51/61 接缝。

## Tasks

### Task 1 — Repair helpers and regression tests (TDD)

先写并运行失败测试，覆盖：

1. repaired ST 坐标选择 `spatial_st_corrected_pre_manual_v4`；
2. feature provenance 为 `copied_unchanged_pre_manual_v4` 时使用 identity；
3. feature provenance 为 `inverse_query_old_field_in_new_v4_frame` 时使用 `inv(M)`；
4. 新输出路径不允许覆盖已有目录。

再实现最小的可测试 helper。

### Task 2 — Materialize repaired anchors and feature stores

读取 v4 candidate H5AD 与 v4 feature/nucleus stores，生成：

- `data/9957_g0/registration_correction_he_v5_coordinate_repair/`
- `data/9957_g0/reconstruction_prep_v5_coordinate_repair/`

在新的 H5AD 中将 repaired coordinate 写入训练使用的 `spatial`、`spatial_registered`、`spatial_st_corrected`，保留旧 coordinate columns。用逆 affine 重新物化 UNI2 与 nucleus stores，并写出矩阵、来源、frame、section provenance manifest。

### Task 3 — Rebuild UOT and morphology/nucleus path cache

使用 repaired anchors、repaired UNI2 store、repaired nucleus store 和新 frame，重新计算 9957/g0 的 sparse top-k UOT 与 morphology/nucleus path cache。禁止复用 v4 cache，运行后检查 metadata 中的 source paths/frame。

### Task 4 — Alignment QC gate

生成新的 ST-only、H&E/ST overlay、section 51→61 接缝和全 section 3D/XY QC 图，输出到 `data/9957_g0/qc/coordinate_repair_v5/`。核对中心位移、mask 覆盖率、frame 名称和 cache 来源。该任务完成后停止，等待用户确认。

### Task 5 — Confirmed training and full reconstruction

仅在用户确认 QC 后，用 repaired anchors/stores/caches 训练 nucleus-path + morphology-UOT + cell-type model，并生成不保留 holdout 的 full-thickness reconstruction。重新生成指标、QC、3D visualization 和完整 provenance。

## Expected validation commands

- `pytest -q tests/test_9957_coordinate_repair.py`
- 相关现有 tests：`tests/test_frame_resample.py`、`tests/test_path_cache_frame_metadata.py`、`tests/test_highres_coordinate_validation.py`
- 修复脚本的 dry-run/manifest validation；
- QC manifest 检查 source frame、target frame、section affine coverage 和 51→61 displacement。

## Out of scope until confirmation

- 训练、ODE sampling、full-thickness reconstruction；
- 修改原 v1/v4 结果；
- 重新进行 STalign 或改变用户已经确认的 H&E 配准。
