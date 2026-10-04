#!/usr/bin/env python3
"""留一系官方 benchmark 推理入口（2026-09-28 定案：启动即 dispatch 分发）。

默认行为（无 --no_dispatch）：守护分发——轮询本机 GPU 显存，空闲卡领单基因
推理任务；单基因结果 part 先落 out_dir/.partial/，全部基因完成后合并成完整
predictions.h5ad 放 out_dir/ 并清空 .partial/。积分步数等运行态参数从
configs/universal.yaml 的 inference 节读取（CLI flag 仍可覆盖）。

用法（远程项目根，先 source .venv）:
  .venv/bin/python -m src.script.inference --checkpoint_path <ckpt>
    # 输出目录固定 = <common.output_base_dir>/inference_<YYYY-MM-DD_HH-MM>（无 --out_dir flag）
  .venv/bin/python -m src.script.inference --no_dispatch --heldout_line=HCT116 \
      [--perts=基因子集]   # 单进程直跑（调试）
  # 复跑/续跑锚定同一输出目录：SCDFM_RUN_TS=<ts> .venv/bin/python -m src.script.inference --checkpoint_path <ckpt>
完成后本地评分：bash scripts/local_benchmark.sh --pred_h5ad <dir>/predictions.h5ad \
    --real_h5ad <dir>/real.h5ad --out_dir <基地址>

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
from pathlib import Path

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
    # n_ctrl_cells/n_pred_cells/n_real_cells/min_real_cells 已提取到 common 节
    # （2026-09-28 定案：build_tensors real 侧三件套与 inference 共用口径，CommonConfig 声明）
    max_perts: int = 0         # 冒烟上限（0=全部）
    seed: int = 42
    de_backend: str = 'pdex'   # 无 gpudge 时显式 CPU DE 后端
    allow_degenerate_baseline: bool = False  # baseline 锚点退化（如 lfc_nmae 显著集 <10 门控）时仍写出
    # 提交侧同款（generate_submission.GenConfig 亦有此二项）
    top_infer_genes: int = field(default=19843, kw_only=True)  # 建模基因数（2026-09-30 定案=19,843 全轴；select_modeled_genes 内 min 到池大小）
    ode_steps: int = 100
    mask_fname: str = ''  # artifact_paths 需要该字段（空=按 split_method/topk 派生）
    # 多卡分片（2026-09-15：单卡串行 286 基因 ODE ~4min/基因太慢，8 卡分片）
    perts: str = ''          # 逗号分隔基因子集；空=全部（配合 no_eval 空串=仅 prep real+perts.txt）
    context: str = ''        # 推理 context（2026-09-30 多 control 源定案）：worker 只取该 context 对照
    no_eval: bool = False    # 只构建 pred 不跑三件套（分片 worker）
    eval_only: bool = False  # 跳过构建：拼接 out_dir/pred*.h5ad + real.h5ad 后跑三件套
    pred_tag: str = ''       # 分片文件名后缀 -> pred_{tag}.h5ad
    reuse_real: bool = False # real.h5ad 已存在则直接读，不重扫语料
    eval_out_dir: str = ''   # eval_only 产物目录（空=out_dir）；部分 eval 用它避免污染最终 scores.csv
    parts_only: bool = False # 只写 .partial 基因级 part，不写整片合并 pred{tag}.h5ad（守护分发单基因 worker）
    # frozen_tensors_dir 继承自 FlowConfig（2026-09-28：原 residual_dir 字段删除，
    # 统一走 YAML common 节，消除与训练侧两套来源的不一致）
    # 分发运行态（2026-09-28 起入 inference 节统一配置）
    free_mb: int = 51200
    poll_s: float = 10.0
    max_retry: int = 3


def _cli_bin() -> str:
    return str(Path(sys.executable).parent / 'cell-eval2')


def _run_cli(args: list[str]) -> None:
    cmd = [_cli_bin()] + args
    print('$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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
            a.obs = derive_pert_columns(a.obs, cfg.obs_col_candidates, cfg.ctrl_sentinels)
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
    a.obs = derive_pert_columns(a.obs, cfg.obs_col_candidates, cfg.ctrl_sentinels)
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
               real: ad.AnnData, device, context: str = '') -> ad.AnnData:
    """从 heldout_line 对照生成扰动预测：ODE → Reŝ → r̂ = r̄_p − ḡ + Reŝ（r̄_c=0）→
    Poisson(ctrl × 2^{r̂}) counts（范式二，2026-09-26）。

    context 非空（2026-09-30 多 control 源定案）：ODE 源=real 中该 context 的对照、
    pred 标签与 Poisson 种子均用该 context；空=heldout_line 旧口径。
    """
    ctx = context or cfg.heldout_line
    ctx_mask = real.obs['context'].astype(str).to_numpy() == ctx
    ctl_idx_all = np.nonzero(
        ((real.obs['target_gene'].astype(str).to_numpy() == 'non-targeting') & ctx_mask))[0]
    ctl_raw = real.X[ctl_idx_all].tocsr()          # raw counts 子矩阵
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
    var_df = pd.DataFrame(index=real.var_names.astype(str))
    for i, pert in enumerate(perts):
        part_path = os.path.join(parts_dir, f'pred{tag}_g{i:03d}.h5ad')
        if os.path.exists(part_path):
            part = ad.read_h5ad(part_path)
            rows.append(part.X.tocsr())
            obs_rows.append(part.obs)
            print(f'pred: reuse part g{i:03d} ({pert}), {len(rows)}/{len(perts)} genes done', flush=True)
            continue
        src_idx = np.sort(rng.choice(len(ctl_idx_all), size=min(cfg.n_pred_cells, len(ctl_idx_all)),
                                     replace=False))
        src_raw = ctl_raw[src_idx]                        # (n, 18533)
        src_norm = ctl_norm[src_idx]                      # log1p 对照
        depths = np.asarray(src_raw.sum(axis=1)).ravel()

        src_modeled = torch.from_numpy(src_norm[:, modeled_pos_full].toarray()).float().to(device)
        pert_id_b = torch.tensor([vocab.encode(pert)], dtype=torch.long,
                                 device=device).repeat(1, 1)
        pred_modeled = ode_predict(
            vf, gene_ids, src_modeled, pert_id_b, cfg.batch_size, cfg.ode_steps,
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
        mean_cts = (src_raw[:, modeled_pos_full].toarray().astype(np.float64)
                    * np.power(2.0, rhat.astype(np.float64)))
        # 全轴恢复（2026-09-27 fix）：建模基因 <- Poisson 采样，非建模基因 <- 对照原样；
        # var 轴 = 全轴 18533，X 必须同宽（旧桥 log1p_bridge_to_counts 同语义）
        counts_full = src_raw.toarray().astype(np.float32)
        counts_full[:, modeled_pos_full] = np.random.default_rng(
            stable_seed(ctx, pert, cfg.seed + 7)).poisson(mean_cts).astype(np.float32)
        obs_g = pd.DataFrame({'target_gene': [pert] * counts_full.shape[0],
                              'context': [ctx] * counts_full.shape[0],
                              'target': [pert] * counts_full.shape[0]})
        rows.append(sparse.csr_matrix(counts_full))
        obs_rows.append(obs_g)
        ad.AnnData(X=sparse.csr_matrix(counts_full, dtype=np.float32), obs=obs_g,
                   var=var_df).write_h5ad(part_path)
        print(f'pred: {len(rows)}/{len(perts)} genes done', flush=True)

    X = sparse.vstack(rows).tocsr()
    obs_df = pd.concat(obs_rows, ignore_index=True)
    pred = ad.AnnData(X=X.astype(np.float32), obs=obs_df,
                      var=pd.DataFrame(index=real.var_names.astype(str)))
    print(f'pred: {X.shape[0]} cells x {X.shape[1]} genes ({len(perts)} genes)', flush=True)
    return pred


def _subsample_pred(pred: ad.AnnData, n: int, seed: int) -> ad.AnnData:
    """每扰动预测细胞抽到 n 个（与 real 侧对齐，DE 功效对称）；对照行原样保留。

    与直接生成 n 个统计等价：pred 的 400 个细胞是同一预测分布的独立样本，
    抽子集 = 同一分布的另一组 n 个样本。400 全量仍留在 pred_shard*.h5ad。
    """
    rng = np.random.default_rng(seed)
    tg = pred.obs['target_gene'].astype(str).to_numpy()
    rows = []
    for p in sorted(set(tg) - {'non-targeting'}):
        idx = np.nonzero(tg == p)[0]
        rows.append(np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)))
    keep = np.concatenate(rows) if rows else np.array([], dtype=int)
    ctl = np.nonzero(tg == 'non-targeting')[0]
    out = pred[np.concatenate([keep, ctl])].copy()
    print(f'eval subsample: {len(rows)} perts x cap {n} -> {len(keep)} pred cells '
          f'+ {len(ctl)} ctl', flush=True)
    return out


def _run_eval(cfg: BenchConfig, real: ad.AnnData, pred: ad.AnnData) -> None:
    """官方三件套：baseline（b）→ run --anchor（u + r 锚点）→ score（s=(u-b)/(r-b)）。"""
    real_path = os.path.join(cfg.out_dir, 'real.h5ad')
    pred_path = os.path.join(cfg.out_dir, 'pred.h5ad')
    pred.write_h5ad(pred_path)  # eval_only 拼接体也落 canonical 名

    base_flags = ['--preset', 'vcc2026', '--input-type', 'counts',
                  '--pert-col', 'target_gene', '--control', 'non-targeting',
                  '--set', f'de.backend={cfg.de_backend}']
    bdir = os.path.join(cfg.out_dir, 'baseline')
    rdir = os.path.join(cfg.out_dir, 'run')
    base_cmd = ['baseline', '-ar', real_path, *base_flags, '-o', bdir]
    if cfg.allow_degenerate_baseline:
        base_cmd.append('--allow-degenerate-baseline')
    _run_cli(base_cmd)
    _run_cli(['run', '-ap', pred_path, '-ar', real_path, *base_flags, '--anchor', '-o', rdir])

    user_agg = os.path.join(rdir, 'agg_results.csv')
    base_agg = os.path.join(bdir, 'baseline_agg.csv')
    # --anchor 要传 anchor 所在目录（其内含 anchor_agg.parquet + anchor_meta.json sidecar），
    # 传文件路径会报 "an anchor directory must carry its sidecar"
    anchor_dir = rdir
    anchor = os.path.join(anchor_dir, 'anchor_agg.parquet')
    if not os.path.exists(anchor):
        raise RuntimeError(
            f'anchor 缺失（{anchor}）：run --anchor 被拒，通常 = 该系真实数据 DE 功效不足，'
            f'无一扰动在 5 折半拆分后通过 lfc_nmae 显著集 ≥10 门控（见 cell_eval2/anchor.py 报错）。'
            f'无法计算复现锚点 r ⇒ 无官方同标度分数。可尝试：提高 min_real_cells/n_ctrl_cells、'
            f'或接受去掉 lfc_nmae 后单独评估其余 5 指标。')
    score_path = os.path.join(cfg.out_dir, 'scores.csv')
    _run_cli(['score', '--user-agg', user_agg, '--baseline-agg', base_agg,
              '--anchor', anchor_dir, '-o', score_path])

    scores = pd.read_csv(score_path)
    print('\n===== vcc2026 scaled scores (s=(u-b)/(r-b), 1 = replicate level) =====', flush=True)
    print(scores.to_string(index=False), flush=True)
    print(f'\nartifacts in {cfg.out_dir}: real.h5ad pred.h5ad baseline/ run/ scores.csv', flush=True)


def main() -> None:
    cfg = ConfigUtils.load(BenchConfig, description=__doc__,
                           extra_sections=('inference',))
    assert cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path), 'checkpoint_path required'
    # 输出目录固定 yaml 派生（2026-09-29 定案）：<output_base_dir>/inference_<ts>，
    # 不接受 --out_dir；复跑锚定同目录用 SCDFM_RUN_TS=<ts>
    cfg.out_dir = cfg.make_path('inference')
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    real_path = os.path.join(cfg.out_dir, 'real.h5ad')

    # ---- eval_only：拼接 pred*.h5ad（或 .partial 基因级 part）+ real.h5ad，直接跑三件套 ----
    if cfg.eval_only:
        real = sc.read_h5ad(real_path)
        parts = sorted(glob.glob(os.path.join(cfg.out_dir, 'pred_shard*.h5ad')))
        if not parts and os.path.exists(os.path.join(cfg.out_dir, 'predictions.h5ad')):
            parts = [os.path.join(cfg.out_dir, 'predictions.h5ad')]  # dispatch 合并产物
        if not parts:
            # 部分 eval（2026-09-23）：整片未跑完时退到每基因落盘的 part，
            # 评已完成基因的初步得分（real 侧收口到 pred 实际覆盖的基因）。
            parts = sorted(glob.glob(os.path.join(cfg.out_dir, '.partial', 'pred_shard*_g*.h5ad')))
            assert parts, f'no pred_shard*.h5ad nor .partial in {cfg.out_dir}'
        preds = [sc.read_h5ad(p) for p in parts]
        pred = ad.concat(preds, join='outer', index_unique=None)
        # real 收口到 pred 覆盖的扰动基因（validate_pair 要求两侧扰动集一致；
        # 全量 eval 时 pred 覆盖全部 300 基因，此过滤为无操作）
        pred_genes = set(pred.obs['target_gene'].astype(str).unique()) - {'non-targeting'}
        tg = real.obs['target_gene'].astype(str).to_numpy()
        real = real[(tg == 'non-targeting') | np.isin(tg, list(pred_genes))].copy()
        print(f'eval_only: {len(parts)} parts, {len(pred_genes)} genes, '
              f'real narrowed to {real.shape[0]} cells', flush=True)
        # 官方口径：pred 侧必须同样含对照类别（non-targeting）。对照本就不预测，
        # 拷贝 real 的对照 counts 补齐，使两侧扰动集合一致（validate_pair 要求逐项相同）。
        ctl_mask = real.obs['target_gene'].values == 'non-targeting'
        ctl = real[ctl_mask].copy()
        pred = ad.concat([pred, ctl], join='outer', index_unique=None)
        print(f'eval_only: concat {len(parts)} parts + {ctl.shape[0]} ctl -> '
              f'{pred.shape[0]} cells x {pred.shape[1]} genes', flush=True)
        # 2026-09-19 用户定案：eval 前 pred 每扰动抽到 n_real_cells(100)，
        # 与 real 侧 100 对齐（DE 检验功效对称），再进三件套
        pred = _subsample_pred(pred, cfg.n_real_cells, cfg.seed)
        if cfg.eval_out_dir:
            os.makedirs(cfg.eval_out_dir, exist_ok=True)
            real.write_h5ad(os.path.join(cfg.eval_out_dir, 'real.h5ad'))
            cfg.out_dir = cfg.eval_out_dir
        _run_eval(cfg, real, pred)
        return

    # ---- real 构建（分片 worker 复用已建好的 real.h5ad）----
    if cfg.reuse_real and os.path.exists(real_path):
        real = sc.read_h5ad(real_path)
        print(f'reuse real.h5ad: {real.shape[0]} cells', flush=True)
    else:
        real = build_real(cfg)
        real.write_h5ad(real_path)

    # ---- prep 模式：只建 real + 写基因清单（不加载模型、不做预测）----
    if cfg.no_eval and not cfg.perts:
        tg = real.obs['target_gene'].astype(str).to_numpy()
        perts = sorted(p for p in set(tg[tg != 'non-targeting']))
        with open(os.path.join(cfg.out_dir, 'perts.txt'), 'w') as f:
            f.write('\n'.join(perts) + '\n')
        print(f'PREP_DONE: real={real.shape[0]} cells, {len(perts)} perts -> perts.txt', flush=True)
        return

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

    vf = instantiate_model(cfg.model_type, ntoken=cfg.ntoken, d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path)
    ckpt = torch.load(cfg.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()

    pred = build_pred(cfg, vf, gene_ids, vocab, modeled, real, device,
                      context=cfg.context)
    if not cfg.parts_only:
        tag = f'_{cfg.pred_tag}' if cfg.pred_tag else ''
        pred_path = os.path.join(cfg.out_dir, f'pred{tag}.h5ad')
        pred.write_h5ad(pred_path)

    if not cfg.no_eval:
        _run_eval(cfg, real, pred)


# ================= dispatch 子命令（2026-09-28 自 bench_dispatch.py 并入，默认入口）=================
# 守护分发：每 poll_s 秒轮询本机 GPU 显存，空闲（可用 > free_mb）的卡立刻领下一个
# panel 基因推理任务（单基因一进程，跑完写 .partial 即退出）。全部完成后合并
# predictions.h5ad 并清空 .partial/。运行态参数（ode_steps/batch_size/free_mb/...）
# 默认来自 universal.yaml inference 节，CLI flag 覆盖。
# 跨机协调（.36/.49 各跑一实例，共享盘同一 out_dir）：单实例锁按 host 分离；基因认领
# = .partial/.claims/<gene> O_EXCL 原子创建（NFS 互斥）；任务编号/日志含 host 前缀。
# 断电/重启安全：predictions.h5ad 已存在 → 直接退出；启动时扫描 .partial 全基因级
# part 自动跳过已完成；>24h 或死 pid claim 清理。
HOST = socket.gethostname().split('.')[0]


def free_mem_mib() -> dict[int, int]:
    """{gpu_idx: free_mib}，解析 nvidia-smi 的 total/used（MiB）。"""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.strip().splitlines():
        idx, tot, used = (int(x.strip()) for x in line.split(','))
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

    ap = argparse.ArgumentParser(description='inference 守护分发（默认入口）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint_path', required=True)
    ap.add_argument('--train_set_paths', nargs='*', default=[],
                    help='训练语料文件列表（传给 worker 派生缓存/mask/vocab 键，须与训练时一致）')
    ap.add_argument('--heldout_line', default=icfg.heldout_line)
    ap.add_argument('--gpus', type=int, default=icfg.gpus)
    ap.add_argument('--ode_steps', type=int, default=icfg.ode_steps)
    ap.add_argument('--batch_size', type=int, default=icfg.batch_size)
    ap.add_argument('--free_mb', type=int, default=icfg.free_mb, help='空闲显存门控（MiB）')
    ap.add_argument('--poll_s', type=float, default=icfg.poll_s, help='轮询周期（秒）')
    ap.add_argument('--max_retry', type=int, default=icfg.max_retry)
    ap.add_argument('--settle_s', type=float, default=0.0,
                    help='启动后先等待 N 秒再扫描 done/下发（供重启时等在飞旧 worker 跑完）')
    args = ap.parse_args()

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
            os.path.join(root, '.venv/bin/python'), '-m',
            'src.script.inference',
            '--no_dispatch',   # 防递归：worker 走单进程直跑路径
            '--checkpoint_path', args.checkpoint_path,
            *(['--train_set_paths', *args.train_set_paths] if args.train_set_paths else []),
            '--heldout_line', args.heldout_line,
            '--context', ctx,
            '--no_eval', '--reuse_real',
            '--batch_size', str(args.batch_size),
            '--ode_steps', str(args.ode_steps),
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
        for g in range(args.gpus):
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
    pred = ad.concat([sc.read_h5ad(p) for p in part_paths],
                     join='outer', index_unique=None)
    # 对照补齐（与旧 eval_only 同口径）：pred 侧附上 real 的 non-targeting 行，
    # 供 validate_pair 要求的两侧扰动集一致
    real_path = os.path.join(out_dir, 'real.h5ad')
    assert os.path.exists(real_path), f'missing {real_path}'
    real = sc.read_h5ad(real_path)
    ctl = real[real.obs['target_gene'].values == 'non-targeting'].copy()
    pred = ad.concat([pred, ctl], join='outer', index_unique=None)
    pred.write_h5ad(merged_path)
    shutil.rmtree(os.path.join(out_dir, '.partial'))
    print(f'[dispatch:{HOST}] 合并完成：{merged_path}（{pred.shape[0]} cells x '
          f'{pred.shape[1]} genes），.partial 已清空', flush=True)
    print(f'[dispatch:{HOST}] 本地评分：bash scripts/local_benchmark.sh '
          f'--pred_h5ad {merged_path} --real_h5ad {real_path} --out_dir {out_dir}',
          flush=True)


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
