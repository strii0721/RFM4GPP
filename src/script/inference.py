#!/usr/bin/env python3
"""留一系官方 benchmark 推理入口（2026-09-28 定案：启动即 dispatch 分发）。

默认行为（无 --no_dispatch）：守护分发——轮询本机 GPU 显存，空闲卡领单基因
推理任务；单基因结果 part 先落 out_dir/.partial/，全部基因完成后合并成完整
predictions.h5ad 放 out_dir/ 并清空 .partial/。积分步数等运行态参数从
configs/universal.yaml 的 inference 节读取（CLI flag 仍可覆盖）。

用法（远程项目根，先 source .venv）:
  .venv/bin/python -m src.script.inference
    # 输出目录固定 = <inference.output_base_dir>/inference_<YYYY-MM-DD_HH-MM>（无 --out_dir flag）
  .venv/bin/python -m src.script.inference --no_dispatch      # 单进程直跑（调试）
  # 配置一律 yaml（2026-10-06 用户定案：不接受配置 flag）；仅任务级参数
  # --perts/--context/--seed/--pred_tag/--parts_only/--reuse_real 供 dispatcher 派发
  # 复跑/续跑锚定同一输出目录：SCDFM_RUN_TS=<ts> .venv/bin/python -m src.script.inference
# 推理只生成预测产物（2026-10-06 用户定案：本地评分不在本入口，官方平台评分）

测试集取自 common.test_set_paths（即使文件含扰动细胞，推理只用其中
扰动为对照组 non-targeting 的细胞作 ODE 源）。
注意：与生成/训练同源派生缓存只读复用；单对 (pert,400 细胞) ODE 共租实测
~2.4-3.5 min，全 300 基因是长任务（tmux 跑）。
"""
import argparse
import glob
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tyro
from scipy import sparse

from src.utils.config_utils import ConfigUtils, FlowConfig
from src.utils.utils import derive_pert_columns
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from tmp.generate_submission import (
    artifact_paths,
    log1p_bridge_to_counts,
    ode_predict,
    select_modeled_genes,
    stable_seed,
)

if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True


@dataclass
class BenchConfig(FlowConfig):
    # 内部输出目录（2026-09-29 定案）：main/dispatch 一律无条件覆盖为
    # make_path('inference')=<output_base_dir>/inference_<ts>，CLI 值被忽略；
    # eval_external_pred 等工具脚本也借此字段注入目录
    out_dir: str = ''
    # main 模型 ckpt（2026-10-06 用户定案：推理三模型分别 yaml 指定——
    # 从 inference 节读；CLI --checkpoint_path 可覆盖）
    checkpoint_path: str = ''
    # n_ctrl_cells/n_pred_cells/n_real_cells/min_real_cells 已提取到 common 节
    # （2026-09-28 定案：build_tensors real 侧三件套与 inference 共用口径，CommonConfig 声明）
    max_perts: int = 0         # 冒烟上限（0=全部）
    seed: int = 42
    # 提交侧同款（generate_submission.GenConfig 亦有此二项）
    top_infer_genes: int = field(default=19843, kw_only=True)  # 建模基因数（2026-09-30 定案=19,843 全轴；select_modeled_genes 内 min 到池大小）
    ode_steps: int = 100
    mask_fname: str = ''  # artifact_paths 需要该字段（空=按 split_method/topk 派生）
    # 多卡分片（2026-09-15：单卡串行 286 基因 ODE ~4min/基因太慢，8 卡分片）
    perts: str = ''          # 逗号分隔基因子集；空=全部
    context: str = ''        # 推理 context（2026-09-30 多 control 源定案）：worker 只取该 context 对照
    # 本地评分字段已删（2026-10-06 用户定案：inference.py 只生成 pred.h5ad，
    # 评分走独立脚本 scripts/local_benchmark.sh 或官方平台）
    pred_tag: str = ''       # 分片文件名后缀 -> pred_{tag}.h5ad
    reuse_real: bool = False # real.h5ad 已存在则直接读，不重扫语料
    parts_only: bool = False # 只写 .partial 基因级 part，不写整片合并 pred{tag}.h5ad（守护分发单基因 worker）
    # frozen_tensors_dir 继承自 FlowConfig（2026-09-28：原 residual_dir 字段删除，
    # 统一走 YAML common 节，消除与训练侧两套来源的不一致）
    # 分发运行态（2026-09-28 起入 inference 节统一配置）
    free_mb: int = 51200
    poll_s: float = 10.0
    max_retry: int = 3
    cuda_devices: str = ''  # dispatcher 可见卡（2026-10-06 用户定案，inference 节配置；
                            # '4,5,6,7' 形式；空=全部可见，dispatch 启动时注入 CUDA_VISIBLE_DEVICES）
    settle_s: float = 0.0   # 启动后先等待 N 秒再扫描 done/下发（重启时等在飞旧 worker 跑完）


def build_real(cfg: BenchConfig, include_perts: bool = True) -> ad.AnnData:
    """heldout_line 真实细胞：对照子采样 + 全部（≥min_real_cells 的）扰动，raw counts。

    include_perts=False：仅对照（2026-09-29 定案，dispatch 用——worker 的 ODE 源
    只需要对照，扰动基因名由 --perts flag 驱动；完整版由 benchmark.sh 构建）。

    test_set_paths 非空（split_method='whole'，replogle 2026-09-17）：测试语料是
    独立文件，整个文件即 heldout_line，不再按 line_col 过滤；
    否则从训练语料中按 line_col == heldout_line 提取（原留一系口径）。
    """
    # 推理 control 多源（2026-09-30 用户定案）：include_perts=False 且配置了
    # inference_control_paths 时，对照逐文件取 target_gene=='non-targeting' 细胞、
    # 保留各文件原生 context 标签（官方 A/B/C 语义）；完整版（benchmark 用，
    # 对照+扰动参考）仍走 test_set_paths 旧口径。
    if not include_perts and cfg.inference_control_paths:
        rng = np.random.default_rng(cfg.seed)
        segs = []
        for p in cfg.inference_control_paths:
            a = sc.read_h5ad(p, backed='r')
            a.obs = derive_pert_columns(a.obs)
            tg = a.obs['target_gene'].astype(str).to_numpy()
            idx = np.nonzero(tg == 'non-targeting')[0]
            sel = np.sort(rng.choice(idx, size=min(cfg.n_ctrl_cells, len(idx)),
                                     replace=False))
            seg = a[sel].to_memory().copy()
            a.file.close()
            segs.append(seg)
        real = ad.concat(segs, join='outer', index_unique=None)
        real.obs['target'] = real.obs['target_gene'].astype(str)
        # 轴对齐到训练轴（2026-09-30：官方 controls 为 18,533 轴、训练轴 19,843；
        # 缺失基因补 0 列，列序按 frozen_tensors_dir/genes_cache.csv）
        axis_genes = pd.read_csv(
            os.path.join(cfg.frozen_tensors_dir, 'genes_cache.csv'))['gene'].tolist()
        real = _align_var_axis(real, axis_genes)
        ctxs = sorted(set(real.obs['context'].astype(str)))
        print(f'real(controls only, {len(cfg.inference_control_paths)} sources): '
              f'{real.shape[0]} cells x {real.shape[1]} genes, contexts={ctxs}',
              flush=True)
        return real

    src_path = cfg.test_set_paths or cfg.train_set_paths[0]
    a = sc.read_h5ad(src_path, backed='r')
    # 新对齐语料：只读派生 target_gene/context 并赋回视图（backed 模式不落盘，
    # 后续切片 a[real_mask] 才能携带派生列）
    a.obs = derive_pert_columns(a.obs)
    obs = a.obs
    if cfg.test_set_paths:
        line_mask = np.ones(a.n_obs, dtype=bool)  # 独立测试文件：全量即该系
    else:
        line_mask = (obs[cfg.line_col].astype(str) == cfg.heldout_line).to_numpy()
    tg = obs['target_gene'].astype(str).to_numpy()
    ctl_mask = line_mask & (tg == 'non-targeting')
    pert_mask = line_mask & (tg != 'non-targeting')

    rng = np.random.default_rng(cfg.seed)
    ctl_idx = np.nonzero(ctl_mask)[0]
    ctl_sel = np.sort(rng.choice(ctl_idx, size=min(cfg.n_ctrl_cells, len(ctl_idx)), replace=False))

    sel_pert = []
    keep_perts: list[str] = []
    n_capped = 0
    if include_perts:
        perts, counts = np.unique(tg[pert_mask], return_counts=True)
        keep_perts = [p for p, c in zip(perts, counts) if c >= cfg.min_real_cells]
        # 基因范围收口到 panel 300（cfg.panel_csv_path = VCC 官方 pert_counts.csv，
        # 用户自选——2026-09-19 实测与官方 VCC panel 零重叠；语料里有而 panel 外的扰动不参与打分）
        panel_raw = pd.read_csv(cfg.panel_csv_path, header=None)[0].astype(str).tolist()
        panel_set = {g for g in panel_raw if g != 'target_gene'}
        keep_perts = [p for p in keep_perts if p in panel_set]
        if cfg.max_perts:
            keep_perts = keep_perts[:cfg.max_perts]

        # 每扰动参考细胞数 = min(n_real_cells, 实际)（2026-09-19 用户定案：官方参考
        # 400/基因，RPE1 天然数据多数基因不足；100 时 panel 300 个全部抽满）
        for p in keep_perts:
            idx_p = np.nonzero((tg == p) & pert_mask)[0]
            n = min(cfg.n_real_cells, len(idx_p))
            if n == cfg.n_real_cells:
                n_capped += 1
            sel_pert.append(np.sort(rng.choice(idx_p, size=n, replace=False)))
    pert_idx = np.concatenate(sel_pert) if sel_pert else np.array([], dtype=int)

    real_mask = np.zeros(a.n_obs, dtype=bool)
    real_mask[ctl_sel] = True
    real_mask[pert_idx] = True
    real = a[real_mask].to_memory().copy()
    real.obs['context'] = cfg.heldout_line
    real.obs['target'] = real.obs['target_gene'].astype(str)
    if include_perts:
        print(f'real: {real.shape[0]} cells = {len(ctl_sel)} ctl + {len(pert_idx)} pert '
              f'({len(keep_perts)} genes, {n_capped} capped at {cfg.n_real_cells})', flush=True)
    else:
        print(f'real(controls only): {real.shape[0]} cells', flush=True)
    return real


def _align_var_axis(real: ad.AnnData, target_genes: list[str]) -> ad.AnnData:
    """var 重排到 target 轴：缺失基因补 0 列、多余列丢弃，列序 = target 序。"""
    pos = {g: i for i, g in enumerate(real.var_names.astype(str))}
    idx = np.array([pos.get(g, -1) for g in target_genes], dtype=np.int64)
    valid = idx >= 0
    Xv = real.X[:, idx[valid]]
    out = sparse.lil_matrix((real.n_obs, len(target_genes)), dtype=real.X.dtype)
    out[:, valid] = Xv
    return ad.AnnData(X=out.tocsr(), obs=real.obs.copy(),
                      var=pd.DataFrame(index=target_genes))


def _norm_log1p(raw: sparse.csr_matrix) -> sparse.csr_matrix:
    """normalize_total(1e4) + log1p，与 data.py / generate_submission 预处理一致。"""
    norm = raw.copy().tocsr()
    totals = np.asarray(norm.sum(axis=1)).ravel()
    norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
    norm.data = np.log1p(norm.data)
    return norm


def build_pred(cfg: BenchConfig, vf, gene_ids, vocab: GeneVocab, modeled: list[str],
               real: ad.AnnData, device, context: str = '', ae=None) -> ad.AnnData:
    """从 heldout_line 对照生成扰动预测：ODE → Reŝ → r̂ = r̄_p − ḡ + Reŝ（r̄_c=0）→
    Poisson(ctrl × 2^{r̂}) counts（范式二，2026-09-26）。

    context 非空（2026-09-30 多 control 源定案）：ODE 源=real 中该 context 的对照、
    pred 标签与 Poisson 种子均用该 context；空=heldout_line 旧口径。
    """
    ctx = context or cfg.heldout_line
    ctx_mask = real.obs['context'].astype(str).to_numpy() == ctx
    ctl_idx_all = np.nonzero(
        ((real.obs['target_gene'].astype(str).to_numpy() == 'non-targeting') & ctx_mask))[0]
    # 兼容：对齐副本曾为 dense（2026-10-06 已转 csr，此处防御）
    ctl_raw = (real.X[ctl_idx_all].tocsr() if hasattr(real.X, 'tocsr')
               else sparse.csr_matrix(real.X[ctl_idx_all]))  # raw counts 子矩阵
    ctl_norm = _norm_log1p(ctl_raw)                # log1p(CP10k)

    # 残差目标常量（范式二）：r̄_p / ḡ，按 modeled 序对齐（r̄_c(RPE1)=0 用户定案）
    rbar_p_all = np.load(os.path.join(cfg.frozen_tensors_dir, 'rbar_p.npy'), mmap_mode='r')
    rbar_p_perts = pd.read_csv(os.path.join(cfg.frozen_tensors_dir, 'rbar_p_perts.csv'))['pert'].tolist()
    gbar_all = np.load(os.path.join(cfg.frozen_tensors_dir, 'gbar.npy'))
    cache_genes = pd.read_csv(os.path.join(cfg.frozen_tensors_dir, 'genes_cache.csv'))['gene'].tolist()
    assert set(cache_genes) == set(modeled), 'residual genes_cache != modeled axis'
    align = np.array([cache_genes.index(g) for g in modeled], dtype=np.int64)
    rbar_p = np.asarray(rbar_p_all[:, align], dtype=np.float32)   # (n_perts, |modeled|)
    gbar = np.asarray(gbar_all[align], dtype=np.float32)

    gene_axis_pos = {g: i for i, g in enumerate(real.var_names)}
    # 对照子矩阵的列轴 = 全轴（real 未做过列过滤）
    modeled_pos_full = np.array([gene_axis_pos[g] for g in modeled], dtype=np.int64)
    # 扰动基因名：worker 由 --perts flag 驱动（2026-09-29 定案——real 为纯对照时
    # 不含扰动列）；单进程直跑无 --perts 时退回 real 实际扰动集
    if cfg.perts:
        perts = [p for p in cfg.perts.split(',') if p]
    else:
        perts = sorted(p for p in real.obs['target_gene'].astype(str).unique() if p != 'non-targeting')
    assert all(p in gene_axis_pos for p in perts), \
        'perturbation target gene missing from real var axis (zero-target invariant broken)'
    rng = np.random.default_rng(cfg.seed)
    rows, obs_rows = [], []
    # 每基因立即落盘（2026-09-23 用户定案）：进程被杀不丢已完成基因；
    # 同片重启自动跳过已存在 part（resume-skip），整片末尾仍合并写
    # pred{tag}.h5ad 供 eval 使用（eval 逻辑不变）。
    parts_dir = os.path.join(cfg.out_dir, '.partial')
    os.makedirs(parts_dir, exist_ok=True)
    tag = f'_{cfg.pred_tag}' if cfg.pred_tag else ''
    # astype(str)：reuse_real 路径 real 为 backed 读入，var 索引是 nullable StringArray，
    # anndata 默认拒绝写（旧轮 real 内存构建=普通 str 无此问题）
    # 轴还原（2026-10-06 定案；2026-10-07 改名 output_genes_csv）：pred 输出轴 = 输出
    # 轴清单（output_genes_csv 指定，real 为训练轴时取子列）。缺基因补零列（训练轴没有的
    # 输出轴基因，如旧 19547 轴的 TIAF1；新 19657 轴下不应发生）
    _real_pos = {g: i for i, g in enumerate(real.var_names.astype(str))}
    # 别名映射已废弃（2026-10-07）：TIAF1 与 MYO18A 是不同基因（前体关系），
    # 新词表 mrnh19657 中 TIAF1 独立成列；缺基因仍走补零列
    if cfg.output_genes_csv:
        _out_genes = pd.read_csv(cfg.output_genes_csv)['gene_name'].astype(str).tolist()
        _axis_pos_raw = np.array([_real_pos.get(g, -1) for g in _out_genes],
                                 dtype=np.int64)
        _axis_missing = _axis_pos_raw < 0
        _axis_pos = _axis_pos_raw[~_axis_missing]
        var_df = pd.DataFrame(index=_out_genes)
    else:
        _axis_pos = None
        _axis_missing = None
        var_df = pd.DataFrame(index=real.var_names.astype(str))
    for i, pert in enumerate(perts):
        _g0 = time.time()
        part_path = os.path.join(parts_dir, f'pred{tag}_g{i:03d}.h5ad')
        if os.path.exists(part_path):
            part = ad.read_h5ad(part_path)
            # 兼容：旧 part 或新版 anndata 读回可能为 dense ndarray（2026-10-06）
            rows.append(part.X.tocsr() if hasattr(part.X, 'tocsr') else sparse.csr_matrix(part.X))
            obs_rows.append(part.obs)
            print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] pred: reuse part g{i:03d} ({pert}), '
                  f'{len(rows)}/{len(perts)} genes done', flush=True)
            continue
        src_idx = np.sort(rng.choice(len(ctl_idx_all), size=min(cfg.n_pred_cells, len(ctl_idx_all)),
                                     replace=False))
        print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] ode: start {pert} '
              f'({i + 1}/{len(perts)}), {cfg.n_pred_cells} cells x {cfg.ode_steps} steps',
              flush=True)
        src_raw = ctl_raw[src_idx]                        # (n, 18533)
        src_norm = ctl_norm[src_idx]                      # log1p 对照
        depths = np.asarray(src_raw.sum(axis=1)).ravel()

        src_modeled = torch.from_numpy(src_norm[:, modeled_pos_full].toarray()).float().to(device)
        pert_id_b = torch.tensor([vocab.encode(pert)], dtype=torch.long,
                                 device=device).repeat(1, 1)
        if ae is not None:
            # AE 模式（2026-10-06）：表达 → enc → 潜空间 ODE → dec → Reŝ（基因空间）
            ae_enc, ae_dec = ae
            with torch.no_grad():
                src_latent, src_skips = ae_enc(src_modeled)
                pred_latent = ode_predict(
                    vf, gene_ids, src_latent, pert_id_b, cfg.batch_size_per_gpu, cfg.ode_steps,
                    cfg.noise_type, getattr(cfg, 'poisson_alpha', 0.8),
                    getattr(cfg, 'poisson_target_sum', 1e4), device,
                    clamp_output=False,
                )
                pred_modeled = ae_dec(pred_latent, src_skips).cpu().numpy()
        else:
            pred_modeled = ode_predict(
                vf, gene_ids, src_modeled, pert_id_b, cfg.batch_size_per_gpu, cfg.ode_steps,
                cfg.noise_type, getattr(cfg, 'poisson_alpha', 0.8),
                getattr(cfg, 'poisson_target_sum', 1e4), device,
                clamp_output=False,  # 残差空间可负，禁止 clamp（范式二）
            ).cpu().numpy()

        # 残差恢复（范式二，2026-09-26）：r̂ = r̄_p − ḡ + Reŝ（r̄_c(RPE1)=0），
        # counts ~ Poisson(ctrl × 2^{r̂})；靶基因 r̂=−inf → 2^{−inf}=0 → KD 语义
        if pert in rbar_p_perts:
            p_row = rbar_p_perts.index(pert)
            rbar_p_row = rbar_p[p_row][None, :]
        else:
            # 2026-10-04 方案 A（用户定案）：零样本扰动（训练语料未扰动过的靶，
            # 如 panel 中 DNTTIP1/EEF1A2/EPHB2/FZD2/MSANTD4/NT5DC1）无 r̄_p 行——
            # 取 0 → r̂ = Reŝ（ḡ≈0 实测 mean -2.6e-8，可忽略），纯残差预测。
            rbar_p_row = np.zeros((1, rbar_p.shape[1]), dtype=np.float32)
        rhat = (rbar_p_row - gbar[None, :]) + pred_modeled   # (n, L)
        tpos = int(np.nonzero(modeled_pos_full == gene_axis_pos[pert])[0][0])
        rhat[:, tpos] = -np.inf
        # 发散值裁剪（2026-10-07 修复 lam value too large）：模型对个别任务输出极端
        # 残差时 2^r̂ 溢出泊松 lam 上限；正常 r̂∈±5（fold 32×），clip ±30 只截发散、零影响
        rhat = np.clip(rhat, -30.0, 30.0)
        mean_cts = (src_raw[:, modeled_pos_full].toarray().astype(np.float64)
                    * np.power(2.0, rhat.astype(np.float64)))
        # 全轴恢复（2026-09-27 fix）：建模基因 <- Poisson 采样，非建模基因 <- 对照原样；
        # var 轴 = 全轴 18533，X 必须同宽（旧桥 log1p_bridge_to_counts 同语义）
        counts_full = src_raw.toarray().astype(np.float32)
        counts_full[:, modeled_pos_full] = np.random.default_rng(
            stable_seed(ctx, pert, cfg.seed + 7)).poisson(mean_cts).astype(np.float32)
        # 每细胞归一化到 CPM 后取整（2026-10-08 用户定案）：残差范式桥无深度锚、模型 Res
        # 偏大时总 counts 爆超 vcc 上限；行归一化到 1e6 总量（CPM）后 floor——vcc 要求 counts
        # 必须是整数，floor 保证每细胞总量 ≤1e6（硬约束）
        _tot = counts_full.sum(axis=1, keepdims=True)
        counts_full = np.floor(counts_full / np.where(_tot == 0, 1.0, _tot) * 1e6)
        # 轴还原 训练轴 → 18533 提交轴（2026-10-06 用户定案）；缺失基因补零列（防御，新轴下应为空）
        if _axis_pos is not None:
            if _axis_missing is not None and _axis_missing.any():
                _sub = counts_full[:, _axis_pos]
                counts_full = np.zeros((_sub.shape[0], len(_out_genes)), dtype=np.float32)
                counts_full[:, ~_axis_missing] = _sub
            else:
                counts_full = counts_full[:, _axis_pos]
        obs_g = pd.DataFrame({'target_gene': [pert] * counts_full.shape[0],
                              'context': [ctx] * counts_full.shape[0],
                              'target': [pert] * counts_full.shape[0]})
        rows.append(sparse.csr_matrix(counts_full))
        obs_rows.append(obs_g)
        ad.AnnData(X=sparse.csr_matrix(counts_full, dtype=np.float32), obs=obs_g,
                   var=var_df).write_h5ad(part_path)
        print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] pred: {len(rows)}/{len(perts)} genes done '
              f'({pert}: {time.time() - _g0:.1f}s)', flush=True)

    X = sparse.vstack(rows).tocsr()
    obs_df = pd.concat(obs_rows, ignore_index=True)
    pred = ad.AnnData(X=X.astype(np.float32), obs=obs_df, var=var_df)
    print(f'pred: {X.shape[0]} cells x {X.shape[1]} genes ({len(perts)} genes)', flush=True)
    return pred


def main() -> None:
    cfg = ConfigUtils.load(BenchConfig, use_cli=False, extra_sections=('inference',))
    # 任务级参数（2026-10-06 用户定案：配置一律 yaml 读取，本入口仅接受 dispatcher
    # 派发的任务标识 flag——perts/context/seed/pred_tag/parts_only/reuse_real）
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--perts', default=cfg.perts)
    ap.add_argument('--context', default=cfg.context)
    ap.add_argument('--seed', type=int, default=cfg.seed)
    ap.add_argument('--pred_tag', default=cfg.pred_tag)
    ap.add_argument('--parts_only', action='store_true')
    ap.add_argument('--reuse_real', action='store_true')
    args, _ = ap.parse_known_args()
    cfg.perts = args.perts
    cfg.context = args.context
    cfg.seed = args.seed
    cfg.pred_tag = args.pred_tag
    cfg.parts_only = cfg.parts_only or args.parts_only
    cfg.reuse_real = cfg.reuse_real or args.reuse_real
    assert cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path), 'checkpoint_path required'
    # 输出目录固定 yaml 派生（2026-09-29 定案）：<output_base_dir>/inference_<ts>，
    # 不接受 --out_dir；复跑锚定同目录用 SCDFM_RUN_TS=<ts>
    cfg.out_dir = cfg.make_path('inference')
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    real_path = os.path.join(cfg.out_dir, 'real.h5ad')

    # ---- real 构建（分片 worker 复用已建好的 real.h5ad）----
    if cfg.reuse_real and os.path.exists(real_path):
        real = sc.read_h5ad(real_path)
        print(f'reuse real.h5ad: {real.shape[0]} cells', flush=True)
    else:
        real = build_real(cfg)
        real.write_h5ad(real_path)

    # ---- 分片子集：只保留本进程负责的扰动（+ 全部对照）----
    if cfg.perts:
        want = set(cfg.perts.split(','))
        tg = real.obs['target_gene'].astype(str).to_numpy()
        keep = (tg == 'non-targeting') | np.isin(tg, list(want))
        if cfg.context:  # 多 control 源（2026-09-30）：只保留该 context 的对照
            keep &= (real.obs['context'].astype(str).to_numpy() == cfg.context)
        real = real[keep].copy()

    cache, mask_path, vocab_path = artifact_paths(cfg)
    vocab = GeneVocab.from_file(vocab_path)
    modeled = select_modeled_genes(cache, cfg.panel_csv_path, cfg.top_infer_genes, vocab,
                                   pool_path=cfg.train_pool_path)
    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    print(f'modeled genes: {len(modeled)}', flush=True)

    # AE 模式（2026-10-06 warm-start + 联合微调）：序列=潜 token、扰动表独立、
    # 共表达 mask 关（潜 token 与基因图尺寸不符）；加载 encoder/decoder。
    _ae_mode = bool(cfg.ae_encoder_ckpt and cfg.ae_decoder_ckpt)
    if _ae_mode:
        from src.models.autoencoder import AEEncoder, AEDecoder
        ae_enc = AEEncoder(cfg.n_top_genes, 8192, cfg.ae_latent_dim).to(device).eval()
        ae_dec = AEDecoder(cfg.n_top_genes, 8192, cfg.ae_latent_dim).to(device).eval()
        ae_enc.load_state_dict(torch.load(cfg.ae_encoder_ckpt, map_location='cpu'))
        ae_dec.load_state_dict(torch.load(cfg.ae_decoder_ckpt, map_location='cpu'))
        gene_ids = torch.arange(cfg.ae_latent_dim, dtype=torch.long, device=device)
        print(f'[AE] inference warm-start loaded latent={cfg.ae_latent_dim}', flush=True)

    vf = instantiate_model(cfg.model_type, ntoken=cfg.resolve_ntoken(), d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path,
                           pert_ntoken=cfg.resolve_pert_ntoken(),
                           use_perturbation_interaction=cfg.use_perturbation_interaction and not _ae_mode)
    # 三模型分别指定（2026-10-06 用户定案）：main=checkpoint_path（目录时 main_best.pt
    # 优先/否则最新 main_*.pt）；encoder/decoder 严格从 yaml 的 ae_encoder_ckpt /
    # ae_decoder_ckpt 读（微调后把新路径填进 yaml 即可，无隐式覆盖）。
    _ckpt_path = cfg.checkpoint_path
    if os.path.isdir(_ckpt_path):
        _best = os.path.join(_ckpt_path, 'main_best.pt')
        _mains = sorted([p for p in glob.glob(os.path.join(_ckpt_path, 'main_*.pt'))
                         if 'best' not in p])
        _ckpt_path = _best if os.path.exists(_best) else (_mains[0] if _mains else _ckpt_path)
    ckpt = torch.load(_ckpt_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()

    pred = build_pred(cfg, vf, gene_ids, vocab, modeled, real, device,
                      context=cfg.context, ae=(ae_enc, ae_dec) if _ae_mode else None)
    if not cfg.parts_only:
        tag = f'_{cfg.pred_tag}' if cfg.pred_tag else ''
        pred_path = os.path.join(cfg.out_dir, f'pred{tag}.h5ad')
        pred.write_h5ad(pred_path)

    # 推理只生成 pred.h5ad（2026-10-06 用户定案）：本地评分已从本入口剥离——
    # 评分走官方平台（submit 链）或独立脚本 scripts/local_benchmark.sh


# ================= dispatch 子命令（2026-09-28 自 bench_dispatch.py 并入，默认入口）=================
# 守护分发：每 poll_s 秒轮询本机 GPU 显存，空闲（可用 > free_mb）的卡立刻领下一个
# panel 基因推理任务（单基因一进程，跑完写 .partial 即退出）。全部完成后合并
# predictions.h5ad 并清空 .partial/。运行态参数（ode_steps/batch_size_per_gpu/free_mb/...）
# 默认来自 universal.yaml inference 节，CLI flag 覆盖。
# 跨机协调（.36/.49 各跑一实例，共享盘同一 out_dir）：单实例锁按 host 分离；基因认领
# = .partial/.claims/<gene> O_EXCL 原子创建（NFS 互斥）；任务编号/日志含 host 前缀。
# 断电/重启安全：predictions.h5ad 已存在 → 直接退出；启动时扫描 .partial 全基因级
# part 自动跳过已完成；>24h 或死 pid claim 清理。
HOST = socket.gethostname().split('.')[0]


def free_mem_mib() -> dict[int, int]:
    """{gpu_idx: free_mib}，解析 nvidia-smi 的 total/used（MiB）。
    尊重 CUDA_VISIBLE_DEVICES（2026-10-06：worker01 的 0-3 卡有幽灵 exclusive 锁，
    dispatcher 启动 env 指定可见卡即可跳过坏卡）。"""
    vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    allowed = {int(x) for x in vis.split(',') if x.strip() != ''} if vis.strip() else None
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.strip().splitlines():
        idx, tot, used = (int(x.strip()) for x in line.split(','))
        if allowed is not None and idx not in allowed:
            continue
        res[idx] = tot - used
    return res


def _load_done(out_dir: str) -> set[tuple[str, str]]:
    """扫描 .partial 全部基因级 part（含旧分片命名），解析已完成 (context, gene) 对。"""
    done = set()
    for p in glob.glob(os.path.join(out_dir, '.partial', '*_g*.h5ad')):
        try:
            a = ad.read_h5ad(p)
            done.add((str(a.obs['context'].iloc[0]), str(a.obs['target_gene'].iloc[0])))
        except Exception as e:
            print(f'[warn] 跳过坏 part {os.path.basename(p)}: {e}', flush=True)
    return done


def _claim_dir(out_dir: str) -> str:
    d = os.path.join(out_dir, '.partial', '.claims')
    os.makedirs(d, exist_ok=True)
    return d


def _claimed_genes(out_dir: str) -> set[str]:
    return {f for f in os.listdir(_claim_dir(out_dir)) if f != '.keep'}


def _claim_gene(out_dir: str, gene: str) -> bool:
    """O_EXCL 原子认领（跨机 NFS 互斥）。"""
    p = os.path.join(_claim_dir(out_dir), gene)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{HOST}:{os.getpid()}:{time.time()}\n'.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_claim(out_dir: str, gene: str) -> None:
    try:
        os.remove(os.path.join(_claim_dir(out_dir), gene))
    except FileNotFoundError:
        pass


def _pid_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def _cleanup_stale_claims(out_dir: str) -> None:
    """启动清理：本机死 pid 的遗留 claim、>24h 的陈旧 claim。"""
    now = time.time()
    for f in list(_claimed_genes(out_dir)):
        p = os.path.join(_claim_dir(out_dir), f)
        try:
            host, pid, ts = open(p).read().strip().split(':')
        except Exception:
            os.remove(p)
            continue
        stale = (now - float(ts)) > 86400
        dead_local = (host == HOST and _pid_dead(int(pid)))
        if stale or dead_local:
            os.remove(p)
            print(f'[claim] 清理陈旧认领 {f} ({host}:{pid})', flush=True)


def dispatch_main() -> None:
    # argparse 默认值来自 universal.yaml（common+flow+inference 节合并，
    # 与 worker 的 ConfigUtils.load 同口径；checkpoint_path 由本 argparse 提供）
    icfg = ConfigUtils.load(BenchConfig, use_cli=False, extra_sections=('inference',))
    # 可见卡配置注入（2026-10-06 用户定案，yaml inference.cuda_devices；
    # 外部 env 显式设置优先，yaml 值次之；free_mem_mib 据此过滤 + worker spawn 单卡映射）
    if icfg.cuda_devices and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = icfg.cuda_devices

    # 配置一律 yaml（2026-10-06 用户定案：dispatch 不接受配置 flag；settle_s 等运行态
    # 也入 yaml inference 节）。任务级参数（context/perts/seed/pred_tag）仍由 spawn 传递。
    args = icfg

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    # 输出目录固定 yaml 派生（2026-09-29 定案）：<output_base_dir>/inference_<ts>，无 --out_dir；
    # ts 用 SCDFM_RUN_TS 锚定（复跑同目录），未设则实时生成并写入环境传给 worker
    ts = os.environ.get('SCDFM_RUN_TS') or time.strftime('%Y-%m-%d_%H-%M')
    os.environ['SCDFM_RUN_TS'] = ts
    out_dir = os.path.abspath(icfg.make_path('inference'))
    # 2026-09-29 修复：out_dir 由 yaml 派生（原 --out_dir 由调用方预建），dispatch 需自建
    os.makedirs(out_dir, exist_ok=True)

    # 已完成：合并产物存在即全部结束（resume 安全）
    merged_path = os.path.join(out_dir, 'predictions.h5ad')
    if os.path.exists(merged_path):
        print(f'[dispatch:{HOST}] {merged_path} 已存在（全部完成），退出', flush=True)
        sys.exit(0)

    # 单实例锁（按 host 分离，跨机两实例各持一把）
    lock = os.path.join(out_dir, f'.dispatch.pid.{HOST}')
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            print(f'[abort] 本机已有 dispatcher 在跑 (pid={pid}, {lock})', flush=True)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            pass
    open(lock, 'w').write(str(os.getpid()))

    if args.settle_s > 0:
        print(f'[dispatch:{HOST}] 等待在飞 worker 完成 {args.settle_s:.0f}s 后再扫描', flush=True)
        time.sleep(args.settle_s)

    _cleanup_stale_claims(out_dir)

    # worker 日志目录 = <log_base>/<out_dir 基名>/dispatch（2026-09-26 用户定案目录规范）
    log_dir = os.path.join(icfg.log_base_dir, os.path.basename(out_dir), 'dispatch')
    os.makedirs(log_dir, exist_ok=True)

    # dispatcher 自身日志（同规范，不依赖启动方 shell 重定向）
    dlog_path = os.path.join(os.path.dirname(log_dir), f'dispatcher_{HOST}.log')
    dlog = open(dlog_path, 'a', buffering=1)
    os.dup2(dlog.fileno(), 1)
    os.dup2(dlog.fileno(), 2)
    sys.stdout = dlog
    sys.stderr = dlog
    print(f'[dispatch:{HOST}] self-log: {dlog_path}', flush=True)

    # real.h5ad 由 dispatcher 一次性构建（2026-09-29 定案：仅对照——worker ODE 源
    # 只需要对照、扰动基因名由 --perts flag 驱动；完整版真实扰动参考由 benchmark.sh
    # 构建）。若 worker 各自并发 build_real 写同一文件会触发 HDF5 锁冲突 BlockingIOError。
    real_path = os.path.join(out_dir, 'real.h5ad')
    if not os.path.exists(real_path):
        print(f'[dispatch:{HOST}] 构建 real.h5ad（仅对照 {icfg.n_ctrl_cells}）...',
              flush=True)
        real = build_real(icfg, include_perts=False)
        real.write_h5ad(real_path)
        print(f'[dispatch:{HOST}] real.h5ad: {real.shape[0]} cells', flush=True)
    else:
        real = sc.read_h5ad(real_path)

    # 扰动基因名单 = yaml common.panel_csv_path（2026-09-29 用户定案：推理范围
    # 由配置唯一决定，取代 out_dir/perts.txt——旧 prep 步骤不再必需）
    panel_raw = pd.read_csv(icfg.panel_csv_path, header=None)[0].astype(str).tolist()
    all_genes = [g for g in panel_raw if g != 'target_gene']
    # 多 control 源（2026-09-30 用户定案）：任务粒度 = (context, gene)；
    # contexts 从已建 real 的 obs.context 派生（数据自带，不猜）
    contexts = sorted(set(real.obs['context'].astype(str)))
    all_tasks = [(c, g) for c in contexts for g in all_genes]
    done = _load_done(out_dir)
    held = _claimed_genes(out_dir)
    queue = [t for t in all_tasks if t not in done and f'{t[0]}__{t[1]}' not in held]
    print(f'[dispatch:{HOST}] {len(queue)} 待推理 / {len(done)} 已完成 / '
          f'{len(held)} 他机认领中（共 {len(all_tasks)}，contexts={contexts}）', flush=True)

    map_path = os.path.join(out_dir, f'dispatch_map_{HOST}.txt')
    task_next = 0
    if os.path.exists(map_path):
        with open(map_path) as f:
            task_next = sum(1 for _ in f)
    map_f = open(map_path, 'a')

    procs: dict[int, tuple[int, str, str, subprocess.Popen]] = {}  # gpu -> (task, ctx, gene, proc)
    retry: dict[str, int] = {}
    gave_up: list[str] = []

    def spawn(g: int, ctx: str, gene: str) -> None:
        nonlocal task_next
        task = task_next
        task_next += 1
        map_f.write(f'{task}\t{ctx}\t{gene}\n')
        map_f.flush()
        log = os.path.join(log_dir, f'{HOST}_g{g:02d}_t{task:03d}_{ctx}_{gene}.log')
        cmd = [
            sys.executable, '-m',   # 2026-10-06：与 dispatcher 同解释器（worker01 上项目
            'src.script.inference',  # .venv 软链指向他机 uv python，断链不可用）
            '--no_dispatch',   # 防递归：worker 走单进程直跑路径
            # 配置一律 worker 自己从 yaml 读（2026-10-06 定案）；仅传任务级参数
            '--context', ctx,
            '--reuse_real',
            '--seed', str(42 + task),
            '--perts', gene,
            '--pred_tag', f'shard_disp_{HOST}_{task}',
            '--parts_only',
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(g)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        proc = subprocess.Popen(cmd, env=env, stdout=open(log, 'w'),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)  # 脱离 dispatcher 进程组：
        # dispatcher/tmux 被杀不连坐 worker，在飞基因跑完照常落盘
        procs[g] = (task, ctx, gene, proc)
        print(f'[dispatch:{HOST}] gpu{g} <- t{task} {ctx}:{gene} '
              f'({time.strftime("%H:%M:%S")})', flush=True)

    while queue or procs:
        free = free_mem_mib()
        # 收割已退出 worker
        for g, (task, ctx, gene, proc) in list(procs.items()):
            if proc.poll() is not None:
                part = os.path.join(out_dir, '.partial',
                                    f'pred_shard_disp_{HOST}_{task}_g000.h5ad')
                ok = os.path.exists(part)
                del procs[g]
                if ok:
                    _release_claim(out_dir, f'{ctx}__{gene}')
                    done.add((ctx, gene))
                    print(f'[dispatch:{HOST}] gpu{g} t{task} {ctx}:{gene} 完成 '
                          f'({len(done)}/{len(done) + len(queue)})', flush=True)
                else:
                    _release_claim(out_dir, f'{ctx}__{gene}')
                    retry[f'{ctx}__{gene}'] = retry.get(f'{ctx}__{gene}', 0) + 1
                    if retry[f'{ctx}__{gene}'] > args.max_retry:
                        gave_up.append(f'{ctx}__{gene}')
                        print(f'[ERROR] {ctx}:{gene} 失败 {retry[f"{ctx}__{gene}"]} 次，放弃 '
                              f'（日志 logs/dispatch/{HOST}_g{g:02d}_t{task:03d}_{ctx}_{gene}.log）',
                              flush=True)
                    else:
                        queue.append((ctx, gene))
                        print(f'[retry:{HOST}] {ctx}:{gene} 失败（第 {retry[f"{ctx}__{gene}"]} 次），重新入队',
                              flush=True)
        # 空闲（可用 > 门控）且无我方进程的卡领任务（跨机 O_EXCL 认领）
        # 遍历 free 的键（2026-10-06：free 已被 CUDA_VISIBLE_DEVICES 过滤，gpus 上限截取）
        for g in sorted(free.keys())[:args.gpus]:
            if g in procs:
                continue
            if free.get(g, 0) > args.free_mb:
                for i in range(len(queue)):
                    ctx, gene = queue[i]
                    if _claim_gene(out_dir, f'{ctx}__{gene}'):
                        del queue[i]
                        spawn(g, ctx, gene)
                        break
        time.sleep(args.poll_s + random.uniform(0, 2.0))  # 跨机错峰：NFS O_EXCL
        # 在极窄竞争窗下偶发双认领（2026-09-24 实测 ACIN1），加随机抖动把两机
        # 的认领时刻错开；偶发重复基因无害（eval 多 400 细胞同模型样本）

    map_f.close()
    os.remove(lock)
    if gave_up:
        print(f'[dispatch:{HOST}] 失败放弃 {len(gave_up)}: {gave_up}', flush=True)

    # 最终合并：跨机 O_EXCL 认领 + 完整性等待（他机仍在飞时轮询到齐再收口）。
    # 全基因 part 合并成 predictions.h5ad 放 out_dir/，随后清空 .partial/。
    merge_claim = os.path.join(out_dir, '.merge.claim')
    try:
        fd = os.open(merge_claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{HOST}:{os.getpid()}:{time.time()}\n'.encode())
        os.close(fd)
    except FileExistsError:
        print(f'[dispatch:{HOST}] 最终合并由他机负责，本机退出', flush=True)
        sys.exit(0)
    while len(_load_done(out_dir)) < len(all_tasks) - len(gave_up):
        print(f'[dispatch:{HOST}] 等待他机收尾：{len(_load_done(out_dir))}/'
              f'{len(all_tasks) - len(gave_up)}', flush=True)
        time.sleep(120)
    print(f'[dispatch:{HOST}] 全部基因完成，合并 .partial -> predictions.h5ad', flush=True)
    part_paths = sorted(glob.glob(os.path.join(out_dir, '.partial', '*_g*.h5ad')))
    assert part_paths, f'no parts in {out_dir}/.partial'
    # 2026-10-07 用户定案：只合并 .partial 里的 part，不再补 real.h5ad 对照行
    # （旧对照补齐是 validate_pair 时代的残留，混入 non-targeting 行会被 vcc prep 拒收）
    pred = ad.concat([sc.read_h5ad(p) for p in part_paths],
                     join='outer', index_unique=None)
    # 裁剪回输出轴清单（part 已同轴，幂等防御；output_genes_csv 为空时保持 part 轴）
    if icfg.output_genes_csv:
        _out_genes = pd.read_csv(icfg.output_genes_csv)['gene_name'].astype(str).tolist()
        pred = pred[:, _out_genes]
    pred.write_h5ad(merged_path)
    shutil.rmtree(os.path.join(out_dir, '.partial'))
    print(f'[dispatch:{HOST}] 合并完成：{merged_path}（{pred.shape[0]} cells x '
          f'{pred.shape[1]} genes），.partial 已清空', flush=True)


if __name__ == '__main__':
    # 默认入口 = 守护分发（2026-09-28 用户定案）；--no_dispatch 走单进程直跑。
    # .venv/bin/python -m src.script.inference --checkpoint_path ... --out_dir ...
    # 单进程直跑（调试）：--no_dispatch --heldout_line=... --perts=基因子集
    if '--dispatch' in sys.argv:
        sys.argv.remove('--dispatch')  # 兼容旧 flag（无 --no_dispatch 时默认即分发）
    if '--no_dispatch' in sys.argv:
        sys.argv.remove('--no_dispatch')
        main()          # 单进程直跑（worker / 调试）
    else:
        dispatch_main() # 默认：守护分发
