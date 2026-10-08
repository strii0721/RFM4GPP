# PC-token 改造实施计划（19,843 基因轴 → 4,096 PC 序列）

**日期**：2026-10-05
**状态**：待用户确认决策点后执行
**前置定案**（本会话逐步讨论确认）：

1. **放弃 mask 共表达图**（用户拍板）——`use_perturbation_interaction=False`，GeneEncoder 只留基因/PC 嵌入
2. **Perceiver 与 PC-token 二选一**——对 ②③ 注意力项等效（M·L≈2.0×10⁷ vs M²≈1.7×10⁷），PC-token 多降 ④（O(L·d²) 稠密算子群）——选定 **PC-token 为主线**（Perceiver 代码留作对照臂）
3. **三段式 ckpt**（Stable-Diffusion 式）：encoder + flow + decoder 独立训练、推理加载
4. **三个冻结张量保持基因表达空间**（r̄_p / r̄_c / ḡ）——Res 链走**链 A**：预测在 PC 空间、decoder 回基因轴后加回
5. **loss 空间**：PCA+MSE 下两空间数学等价（正交投影，差常数项）→ 阶段 1 选 **PC 空间 MSE**（快）；阶段 2 若上 Poisson/可训 decoder 切基因空间
6. **阶段 1 用固定 PCA（零训练成本）**；学习式 encoder/decoder 为阶段 2（若保真不足）

---

## 架构（阶段 1）

```
训练：
  缓存细胞 (log1p, 19,843) ──投影 (x−μ)@V──▶ (4,096)
  res_<line>.npy (19,843)  ──预投影 @V────▶ res_pc_<line>.npy (4,096)   [一次性]
  主模型（原差分块，序列=4,096）→ 输出 Reŝ_pc
  loss = MSE(Reŝ_pc, res_pc)（PC 空间）

推理：
  对照细胞 h5ad → norm_log1p → 投影 → (4,096) → ODE(100 步) → Reŝ_pc
  → decoder 回投影：Reŝ_gene = Reŝ_pc @ Vᵀ   [不加 μ：预测的是 Res 贡献]
  → r̂ = (r̄_p − ḡ) + Reŝ_gene → Poisson 恢复 → crop 18,533 → 提交
  （链条其余部分与现版完全一致）
```

**投影公式（已核实 npz 键）**：`output/pca_allaxis_4096.npz` → `components (19843,4096) float32`、`mean (19843,)`、`std (19843,)`（v4 口径：log1p + 均值中心化，std 仅元数据不用）
- 前向：`z = (x_log1p − mean) @ components`
- 回投影：`x̂ = z @ components.T + mean`（表达空间）；Res 空间回投影**不加 mean**（待验证点，见风险）

---

## 关键代码事实（插入点，已核查）

| 位置 | 现状 | PC 改造 |
|---|---|---|
| `data.py:888` `PerturbationDataset.__getitem__` | 返回 `src_cell_data`/`tgt_cell_data` (B,19843) numpy | 保持不动（投影放 GPU 侧，见 T2） |
| `train.py:90` `train_step` | 输入 source/target/res_target (B,19843) | 入口处投影：`source/target → (B,4096)` |
| `train.py:362` `_res_tables` | 读 `res_<line>.npy`（19,843 列）| 改读 `res_pc_<line>.npy`（4,096 列，预投影产物） |
| `inference.py:282` `src_modeled` | `src_norm[:, modeled_pos_full]` → (B,19843) | 改：`src_norm`（全轴）→投影→ (B,4096) |
| `inference.py:285` `ode_predict` 输出 | pred_modeled (B,19843) | 输出 (B,4096) → 回投影成 (B,19843) 后再进 :302 加回公式 |
| `instantiate_model` 调用点 | `ntoken=19847`（vocab 大小）| PC 模式：`ntoken=4096+specials`（4099）；`use_perturbation_interaction=False` |
| `configs/universal.yaml` | `d_model:512`、`batch_total:16`、`num_latents:1024` | 新增 PC 相关配置（见 T1） |

**模型本体零改动**：输入形状驱动全部维度（gene_id/cell_1/cell_2 从 (B,19843)→(B,4096)），差分块/adaLN/输出头同构。

---

## 分阶段任务与验收

### 阶段 0：材料准备

**T0.1 目标预投影脚本**（新 `tmp/preproject_targets.py`）
- 读 `frozen_tensors_dir/res_<line>.npy` (n_perts,19843) × 3 line → `@ components` → 存 `res_pc_<line>.npy` (n_perts,4096)（新目录 `frozen_tensors_dir/pc4096/`，**不动原产物**）
- 同时投影 `rbar_p.npy`、`rbar_c.npy`、`gbar.npy` 存副本（备用/核对——推理加回在基因轴，理论上不需要 PC 版，留作验证）
- 验收：随机抽 100 行，核对 `res_pc ≈ res @ V` 数值一致；均值 ≈ 0（验证 Res 直接投影无需减均值）

### 阶段 1：投影工具 + 模型侧打通

**T1.1 新建 `src/utils/pca_proj.py`**
- `load_pca(path)` / `project(x, mean, comp)` / `inverse(z, mean, comp)`（numpy + torch 双接口）
- 验收：单测脚本——对照细胞重建 `x→z→x̂`，per-gene Pearson 中位 ≈0.68（复现 v4 保真度）

**T1.2 配置项**（`configs/universal.yaml` + `src/utils/config_utils.py`）
- 新增（暂定名，可调）：`pca_path`、`n_pcs: 4096`、`use_perturbation_interaction: false`、`pc_ntoken`
- 验收：`tmp/test_yaml_config.py` 断言同步 ALL PASS

**T1.3 模型冒烟**（新 `tmp/pc_smoke.py`）
- 构造：`instantiate_model('origin', ntoken=4099, d_model=512, fusion_method='differential_perceiver', use_perturbation_interaction=False)`，输入 (B,4096)
- 验收：forward/backward ✓、输出 (B,4096)、报告 ms/次 与显存（**预期 vs 现 127ms/步 再降一个量级**）

### 阶段 2：训练链改造

**T2.1 `train.py` 投影 + 目标换源**
- `train_step` 入口：`source/target` 经 GPU 投影（comp/mean 作 buffer 常驻——`register_buffer` 或模块级 tensor）
- `_res_tables` 改读 `pc4096/res_pc_<line>.npy`
- 传参：`ntoken`、`use_perturbation_interaction=False`
- 验收：60 步冒烟——loss 正常下降（对照全轴版同区间）、s/it 报告、无 NaN

**T2.2 显存/速度调参**
- 4096 序列下重测 batch：从 batch_total=16 起步，向上探（B=8/卡→16/卡）
- 验收：记录无 OOM 的最大 batch 与 s/it；更新 yaml

**T2.3 全量训练**
- 10,000 步（沿用），产出 `output/train_pc4096_<ts>/`
- 验收：loss 曲线收敛、checkpoint 落盘

### 阶段 3：推理链改造

**T3.1 `inference.py` 输入投影 + 输出回投影**
- `build_pred`：`src_norm`（全轴 19,843）→ 投影 → `ode_predict`；输出回投影 `Reŝ_gene = Reŝ_pc @ Vᵀ`
- 加回公式（:302）与 Poisson 链（:310）保持原样（基因轴）
- 验收：单任务端到端（1 个扰动）——产物 real.h5ad 形状/语义与现版一致（19,843 轴）

**T3.2 未见面扰动方案 A 兼容检查**
- `rbar_p_row = zeros` 路径不受影响（基因轴操作）✓
- 验收：6 个未见基因（DNTTIP1 等）单跑通

**T3.3 全量推理**（900 任务、8 worker、batch 重测）
- 验收：完成 + md5 核对 + `crop_to_vcc_axis` 提交链完整

### 阶段 4：benchmark 对照（科学裁决）

- 同口径评测 **PC-4096 版 vs 全轴 19,843 版**（16/40/48 三档、RE/NMAE/JAC、overall）
- 产出对照表 → 决定"PC 压缩损失"是否可接受
- 若不可接受 → 阶段 5（学习式编解码）或有条件回退

### 阶段 5（条件触发）：学习式 encoder/decoder

- 触发条件：阶段 4 显示 PC 压缩损失显著（尤其 DEG 响应类指标）
- 架构：MLP autoencoder（19,843→4,096→19,843），独立训练两 ckpt；loss 切基因空间（decoder 参与）
- 本计划不展开（待阶段 4 数据）

---

## 文件改动清单（汇总）

| 文件 | 性质 | 改动 |
|---|---|---|
| `configs/universal.yaml` | 配置 | 新增 `pca_path`/`n_pcs`/`use_perturbation_interaction` |
| `src/utils/config_utils.py` | 代码 | 字段同步（同 num_latents 先例） |
| `src/utils/pca_proj.py` | **新建** | 投影/回投影工具（numpy+torch） |
| `src/script/train.py` | 代码 | 入口投影、目标换源、传参 |
| `src/models/instantiate_model.py` | 代码 | `use_perturbation_interaction` 透传 |
| `src/script/inference.py` | 代码 | 输入投影、输出回投影 |
| `tmp/preproject_targets.py` | **新建** | res_*.npy → pc4096 版 |
| `tmp/pc_smoke.py`、`tmp/pc_train_smoke.py` | **新建** | 冒烟/验收脚本 |
| `tmp/test_yaml_config.py` | 测试 | 断言同步 |

**不改**：`src/models/origin/*`（模型本体零改动）、`tmp/generate_submission.py`（推理产物保持基因轴）、缓存/原冻结张量（只增不删）

---

## 决策点（待确认）

| # | 决策 | 建议 |
|---|---|---|
| 1 | PC 维度 q | **4096**（保真 r=0.68 已验证；1024/2048 损失大，且 4096² 注意力已足够便宜） |
| 2 | 投影执行位置 | **GPU 侧**（train_step/推理入口，计算可忽略）——不动 dataset/缓存 |
| 3 | loss 空间 | **PC 空间 MSE**（数学等价 + 快）——阶段 4 后视需要切基因空间+Poisson |
| 4 | 双模式开关 | **保留 gene/pc 双模式**（配置切换）——便于阶段 4 对照实验；不做死 |
| 5 | batch_total | 先冒烟测（4096 序列预期可大幅加大，建议从 16 起向上探） |
| 6 | 训练步数 | 沿用 10,000 |

## 风险与验证点

1. **科学风险（最大）**：PC 压缩（54% 方差丢弃）对**动态预测/扰动响应**的影响未知——Res 的动态成分可能集中也可能分散在剩余子空间——**阶段 4 对照表是最终裁决**；回退路径=保留全轴版 checkpoint
2. **Res 投影细节**：`res_gene @ V`（不减 mean）的正确性——T0.1 验收包含"投影后均值≈0、重建一致性"检查；若发现系统性偏移，备选 `(res − mean_res) @ V` 并记录 mean_res
3. **回投影位置**：推理链中加回公式（`r̄_p − ḡ`）在基因轴——**decoder 输出只能是 Reŝ**（不能是表达绝对值）——代码审查时重点核对
4. **性能预期偏差**：4096 序列的实测时间/显存可能与线性外推有差（④ 项占比上升后 kernel 效率变化）——T1.3/T2.2 用实测数字说话
5. **新轴兼容**：本计划基于现 19,843 轴；新数据集 19,547 轴到位后**同流程重做**（PCA 重拟合、目标重投影、重训）——脚本已参数化，换轴成本 = 重跑阶段 0-3

## 附：与 Perceiver 分支的关系

- Perceiver 改动（`LatentPerceiverBlock`、`perceiver_latent`）**保留在代码库**，作为 ②③ 的备选实现
- PC-token 阶段 1 用**原版差分块**（4,096² 完全负担得起，无需 latent）
- 若阶段 4 显示"保序列 + 保先验"路线需要复活，Perceiver 分支即插即用
