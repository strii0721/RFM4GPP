#!/usr/bin/env python3
"""留一系官方 benchmark（cell-eval2 vcc2026 preset，2026-09-14）。

heldout_line（默认 HCT116）整体留出：
  real = 该系真实细胞 raw counts（扰动全量 + 对照子采样 n_ctrl_cells）
  pred = 模型从该系对照生成：normalize(1e4)+log1p → 单槽条件 ODE → log1p 桥 → counts
然后跑官方三件套：
  1) baseline  -ar real --preset vcc2026            -> baseline_agg.csv（b，mean-response）
  2) run       -ap pred -ar real --anchor           -> agg_results.csv（u）+ anchor_agg.parquet（r 锚点）
  3) score     --user-agg u --baseline-agg b --anchor r -> s=(u-b)/(r-b)，官方榜单同标度
所有 CLI 显式 --set de.backend=pdex（远程无 gpudge GPU 后端，auto 会拒绝静默回退）。

用法（远程项目根，先 source .venv）：
  .venv/bin/python -u src/script/benchmark_line_holdout.py \
      --checkpoint_path output/train/<ts>/iteration_N/checkpoint.pt \
      --split_method=single_line --heldout_line=HCT116 \
      --out_dir output/benchmark/<tag> [--max_perts=20]
注意：与生成/训练同源派生缓存（build_vcc_cache 产物）只读复用；
单对 (pert,400 细胞) ODE 共租实测 ~2.4-3.5 min，全 300 基因是长任务（tmux 跑）。
"""
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tyro
from scipy import sparse

from config.config_flow import FlowConfig
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.script.generate_submission import (
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
    out_dir: str = ''
    n_ctrl_cells: int = 4000   # real 侧对照细胞数（DE 参考组；官方 context=18400，取子集控时长）
    n_pred_cells: int = 400    # 每扰动预测细胞数（官方 400）
    n_real_cells: int = 100    # real 侧每扰动参考细胞数上限（2026-09-19 用户定案：
    # 官方 400 在 RPE1 天然数据上不可行——panel 300 全 ≥101 恰好抽满 100；
    # 不足 100 的基因用全部真实细胞）
    min_real_cells: int = 20   # real 侧每扰动最少细胞数（低于则跳过该基因）
    max_perts: int = 0         # 冒烟上限（0=全部）
    seed: int = 42
    de_backend: str = 'pdex'   # 无 gpudge 时显式 CPU DE 后端
    allow_degenerate_baseline: bool = False  # baseline 锚点退化（如 lfc_nmae 显著集 <10 门控）时仍写出
    # 提交侧同款（generate_submission.GenConfig 亦有此二项）
    top_infer_genes: int = 11919  # 建模基因数（2026-09-20 定案=全轴；select_modeled_genes 内 min 到池大小）
    ode_steps: int = 100
    mask_fname: str = ''  # artifact_paths 需要该字段（空=按 split_method/topk 派生）
    # 多卡分片（2026-09-15：单卡串行 286 基因 ODE ~4min/基因太慢，8 卡分片）
    perts: str = ''          # 逗号分隔基因子集；空=全部（配合 no_eval 空串=仅 prep real+perts.txt）
    no_eval: bool = False    # 只构建 pred 不跑三件套（分片 worker）
    eval_only: bool = False  # 跳过构建：拼接 out_dir/pred*.h5ad + real.h5ad 后跑三件套
    pred_tag: str = ''       # 分片文件名后缀 -> pred_{tag}.h5ad
    reuse_real: bool = False # real.h5ad 已存在则直接读，不重扫语料
    eval_out_dir: str = ''   # eval_only 产物目录（空=out_dir）；部分 eval 用它避免污染最终 scores.csv
    parts_only: bool = False # 只写 predparts 基因级 part，不写整片合并 pred{tag}.h5ad（守护分发单基因 worker）
    residual_dir: str = 'tmp/residual_targets'  # 范式二常量（rbar_p/gbar/genes_cache；中间缓存，2026-09-26）


def _cli_bin() -> str:
    return str(Path(sys.executable).parent / 'cell-eval2')


def _run_cli(args: list[str]) -> None:
    cmd = [_cli_bin()] + args
    print('$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_real(cfg: BenchConfig) -> ad.AnnData:
    """heldout_line 真实细胞：对照子采样 + 全部（≥min_real_cells 的）扰动，raw counts。

    test_corpus_path 非空（split_method='whole'，replogle 2026-09-17）：测试语料是
    独立文件，整个文件即 heldout_line，不再按 line_col 过滤；
    否则从训练语料中按 line_col == heldout_line 提取（原留一系口径）。
    """
    src_path = cfg.test_corpus_path or cfg.corpus_path
    a = sc.read_h5ad(src_path, backed='r')
    obs = a.obs
    if cfg.test_corpus_path:
        line_mask = np.ones(a.n_obs, dtype=bool)  # 独立测试文件：全量即该系
    else:
        line_mask = (obs[cfg.line_col].astype(str) == cfg.heldout_line).to_numpy()
    tg = obs['target_gene'].astype(str).to_numpy()
    ctl_mask = line_mask & (tg == 'non-targeting')
    pert_mask = line_mask & (tg != 'non-targeting')

    rng = np.random.default_rng(cfg.seed)
    ctl_idx = np.nonzero(ctl_mask)[0]
    ctl_sel = np.sort(rng.choice(ctl_idx, size=min(cfg.n_ctrl_cells, len(ctl_idx)), replace=False))

    perts, counts = np.unique(tg[pert_mask], return_counts=True)
    keep_perts = [p for p, c in zip(perts, counts) if c >= cfg.min_real_cells]
    # 基因范围收口到 panel 300（cfg.panel_path = REPLOGLE_DATA_DIR/pert_counts.csv，
    # 用户自选——2026-09-19 实测与官方 VCC panel 零重叠；语料里有而 panel 外的扰动不参与打分）
    panel_raw = pd.read_csv(cfg.panel_path, header=None)[0].astype(str).tolist()
    panel_set = {g for g in panel_raw if g != 'target_gene'}
    keep_perts = [p for p in keep_perts if p in panel_set]
    if cfg.max_perts:
        keep_perts = keep_perts[:cfg.max_perts]

    # 每扰动参考细胞数 = min(n_real_cells, 实际)（2026-09-19 用户定案：官方参考
    # 400/基因，RPE1 天然数据多数基因不足；100 时 panel 300 个全部抽满）
    sel_pert = []
    n_capped = 0
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
    print(f'real: {real.shape[0]} cells = {len(ctl_sel)} ctl + {len(pert_idx)} pert '
          f'({len(keep_perts)} genes, {n_capped} capped at {cfg.n_real_cells})', flush=True)
    return real


def _norm_log1p(raw: sparse.csr_matrix) -> sparse.csr_matrix:
    """normalize_total(1e4) + log1p，与 data.py / generate_submission 预处理一致。"""
    norm = raw.copy().tocsr()
    totals = np.asarray(norm.sum(axis=1)).ravel()
    norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
    norm.data = np.log1p(norm.data)
    return norm


def build_pred(cfg: BenchConfig, vf, gene_ids, vocab: GeneVocab, modeled: list[str],
               real: ad.AnnData, device) -> ad.AnnData:
    """从 heldout_line 对照生成扰动预测：ODE → Reŝ → r̂ = r̄_p − ḡ + Reŝ（r̄_c=0）→
    Poisson(ctrl × 2^{r̂}) counts（范式二，2026-09-26）。"""
    ctl_idx_all = np.nonzero((real.obs['target_gene'].astype(str) == 'non-targeting').to_numpy())[0]
    ctl_raw = real.X[ctl_idx_all].tocsr()          # raw counts 子矩阵
    ctl_norm = _norm_log1p(ctl_raw)                # log1p(CP10k)

    # 残差目标常量（范式二）：r̄_p / ḡ，按 modeled 序对齐（r̄_c(RPE1)=0 用户定案）
    rbar_p_all = np.load(os.path.join(cfg.residual_dir, 'rbar_p.npy'), mmap_mode='r')
    rbar_p_perts = pd.read_csv(os.path.join(cfg.residual_dir, 'rbar_p_perts.csv'))['pert'].tolist()
    gbar_all = np.load(os.path.join(cfg.residual_dir, 'gbar.npy'))
    cache_genes = pd.read_csv(os.path.join(cfg.residual_dir, 'genes_cache.csv'))['gene'].tolist()
    assert set(cache_genes) == set(modeled), 'residual genes_cache != modeled axis'
    align = np.array([cache_genes.index(g) for g in modeled], dtype=np.int64)
    rbar_p = np.asarray(rbar_p_all[:, align], dtype=np.float32)   # (n_perts, |modeled|)
    gbar = np.asarray(gbar_all[align], dtype=np.float32)

    gene_axis_pos = {g: i for i, g in enumerate(real.var_names)}
    # 对照子矩阵的列轴 = 全轴（real 未做过列过滤）
    modeled_pos_full = np.array([gene_axis_pos[g] for g in modeled], dtype=np.int64)
    assert all(p in gene_axis_pos for p in
               set(real.obs['target_gene'].astype(str)) - {'non-targeting'}), \
        'perturbation target gene missing from real var axis (zero-target invariant broken)'

    perts = sorted(p for p in real.obs['target_gene'].astype(str).unique() if p != 'non-targeting')
    rng = np.random.default_rng(cfg.seed)
    rows, obs_rows = [], []
    # 每基因立即落盘（2026-09-23 用户定案）：进程被杀不丢已完成基因；
    # 同片重启自动跳过已存在 part（resume-skip），整片末尾仍合并写
    # pred{tag}.h5ad 供 eval 使用（eval 逻辑不变）。
    parts_dir = os.path.join(cfg.out_dir, 'predparts')
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
        assert pert in rbar_p_perts, f'{pert} missing from training rbar_p'
        p_row = rbar_p_perts.index(pert)
        rhat = (rbar_p[p_row][None, :] - gbar[None, :]) + pred_modeled   # (n, L)
        tpos = int(np.nonzero(modeled_pos_full == gene_axis_pos[pert])[0][0])
        rhat[:, tpos] = -np.inf
        mean_cts = (src_raw[:, modeled_pos_full].toarray().astype(np.float64)
                    * np.power(2.0, rhat.astype(np.float64)))
        # 全轴恢复（2026-09-27 fix）：建模基因 <- Poisson 采样，非建模基因 <- 对照原样；
        # var 轴 = 全轴 11919，X 必须同宽（旧桥 log1p_bridge_to_counts 同语义）
        counts_full = src_raw.toarray().astype(np.float32)
        counts_full[:, modeled_pos_full] = np.random.default_rng(
            stable_seed(cfg.heldout_line, pert, cfg.seed + 7)).poisson(mean_cts).astype(np.float32)
        obs_g = pd.DataFrame({'target_gene': [pert] * counts_full.shape[0],
                              'context': [cfg.heldout_line] * counts_full.shape[0],
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
    cfg = tyro.cli(BenchConfig, description=__doc__)
    assert cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path), 'checkpoint_path required'
    assert cfg.out_dir, '--out_dir required'
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    real_path = os.path.join(cfg.out_dir, 'real.h5ad')

    # ---- eval_only：拼接 pred*.h5ad（或 predparts 基因级 part）+ real.h5ad，直接跑三件套 ----
    if cfg.eval_only:
        real = sc.read_h5ad(real_path)
        parts = sorted(glob.glob(os.path.join(cfg.out_dir, 'pred_shard*.h5ad')))
        if not parts:
            # 部分 eval（2026-09-23）：整片未跑完时退到每基因落盘的 part，
            # 评已完成基因的初步得分（real 侧收口到 pred 实际覆盖的基因）。
            parts = sorted(glob.glob(os.path.join(cfg.out_dir, 'predparts', 'pred_shard*_g*.h5ad')))
            assert parts, f'no pred_shard*.h5ad nor predparts in {cfg.out_dir}'
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
        real = real[keep].copy()

    cache, mask_path, vocab_path = artifact_paths(cfg)
    vocab = GeneVocab.from_file(vocab_path)
    modeled = select_modeled_genes(cache, cfg.panel_path, cfg.top_infer_genes, vocab,
                                   pool_path=cfg.train_pool_path)
    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    print(f'modeled genes: {len(modeled)}', flush=True)

    vf = instantiate_model(cfg.model_type, ntoken=cfg.ntoken, d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path)
    ckpt = torch.load(cfg.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()

    pred = build_pred(cfg, vf, gene_ids, vocab, modeled, real, device)
    if not cfg.parts_only:
        tag = f'_{cfg.pred_tag}' if cfg.pred_tag else ''
        pred_path = os.path.join(cfg.out_dir, f'pred{tag}.h5ad')
        pred.write_h5ad(pred_path)

    if not cfg.no_eval:
        _run_eval(cfg, real, pred)


if __name__ == '__main__':
    main()
