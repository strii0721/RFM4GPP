#!/usr/bin/env python3
"""检验"条件混合"假设：模型预测效应是否跨系不敏感（2026-09-15）。

两训练内系 HEK293T vs K562：对同一扰动，各自对照出发跑 ODE 得预测 Δ̂，
与真实数据伪bulk效应 Δ_real 做跨系相关对比：
  corr(Δ̂_HEK, Δ̂_K562)  vs  corr(Δ_real_HEK, Δ_real_K562)   （log1p 空间、建模基因轴）
判定：若模型侧跨系相关显著高于真实侧 → 模型把系间效应差异抹平（假设成立）。
"""
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, '/ssd1/ict2/Projects/scDFM')

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tyro
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from src.utils.config_utils import FlowConfig
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from tmp.generate_submission import artifact_paths, ode_predict, select_modeled_genes

if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


@dataclass
class Cfg(FlowConfig):
    checkpoint_path: str = ''   # 2026-10-06：FlowConfig 字段删除后本地自带（CLI 传）
    out_dir: str = '/ssd1/ict2/Projects/vcc-2026/output/line_cond_test'
    line1: str = 'HEK293T'
    line2: str = 'K562'
    n_ctl_model: int = 100     # 模型输入对照细胞数（1 batch）
    n_ctl_real: int = 300      # 真实基线对照细胞数
    n_pert_real: int = 300     # 真实扰动细胞数（伪bulk）
    max_perts: int = 30        # 共享扰动基因数上限
    min_cells: int = 100       # 两系各自最少扰动细胞数门槛
    seed: int = 42
    batch_size: int = 128
    ode_steps: int = 100
    mask_fname: str = ''  # artifact_paths 需要（空=按 split_method/topk 派生）
    top_infer_genes: int = 1000  # select_modeled_genes 需要


def norm_log1p(raw: sparse.csr_matrix) -> np.ndarray:
    """raw counts -> log1p(CP10k) 稠密。"""
    norm = raw.copy().tocsr().astype(np.float32)
    totals = np.asarray(norm.sum(axis=1)).ravel()
    norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
    norm.data = np.log1p(norm.data)
    return norm.toarray()


def main():
    cfg = tyro.cli(Cfg, description=__doc__)
    assert os.path.exists(cfg.checkpoint_path), 'checkpoint_path required'
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ---- 模型与基因集 ----
    cache, mask_path, vocab_path = artifact_paths(cfg)
    vocab = GeneVocab.from_file(vocab_path)
    modeled = select_modeled_genes(cache, cfg.panel_csv_path, cfg.top_infer_genes, vocab)
    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    log(f'modeled genes: {len(modeled)}')

    vf = instantiate_model(cfg.model_type, ntoken=cfg.ntoken, d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path)
    ckpt = torch.load(cfg.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()
    log('model loaded')

    # ---- 语料（backed，obs 全内存）----
    a = sc.read_h5ad(cfg.corpus_path, backed='r')
    obs = a.obs
    tg = obs['target_gene'].astype(str).to_numpy()
    ln = obs[cfg.line_col].astype(str).to_numpy()
    var_names = list(a.var_names)
    modeled_pos = np.array([var_names.index(g) for g in modeled], dtype=np.int64)
    log(f'corpus: {a.n_obs} cells, var={len(var_names)}')

    def ctl_idx(line):
        return np.nonzero((ln == line) & (tg == 'non-targeting'))[0]

    def pert_idx(line, p):
        return np.nonzero((ln == line) & (tg == p))[0]

    # 共享扰动：两系各自 >= min_cells
    shared = []
    tg_line1 = set(tg[ln == cfg.line1])
    tg_line2 = set(tg[ln == cfg.line2])
    for p in sorted(tg_line1 & tg_line2):
        if p == 'non-targeting':
            continue
        if len(pert_idx(cfg.line1, p)) >= cfg.min_cells and len(pert_idx(cfg.line2, p)) >= cfg.min_cells:
            shared.append(p)
    shared = shared[:cfg.max_perts]
    log(f'shared perturbations: {len(shared)} (min_cells={cfg.min_cells})')
    log(f'  {shared}')

    ctl1, ctl2 = ctl_idx(cfg.line1), ctl_idx(cfg.line2)
    log(f'ctl pools: {cfg.line1}={len(ctl1)}, {cfg.line2}={len(ctl2)}')

    def delta_from_rows(rows):
        mask = np.zeros(a.n_obs, dtype=bool)
        mask[rows] = True
        sub = a[mask].to_memory()
        X = norm_log1p(sub.X)
        return X[:, modeled_pos]

    def real_delta(line, p, ctl_rows):
        pidx = np.sort(rng.choice(pert_idx(line, p), size=min(cfg.n_pert_real, len(pert_idx(line, p))),
                                  replace=False))
        Xp = delta_from_rows(pidx)
        Xc = delta_from_rows(ctl_rows)
        return Xp.mean(0) - Xc.mean(0)

    rows_out = []
    all_deltas = {}
    for pi, p in enumerate(shared):
        # 各系对照采样：模型输入用前 n_ctl_model，真实基线用全部 n_ctl_real
        c1 = np.sort(rng.choice(ctl1, size=min(cfg.n_ctl_real, len(ctl1)), replace=False))
        c2 = np.sort(rng.choice(ctl2, size=min(cfg.n_ctl_real, len(ctl2)), replace=False))
        deltas_hat = {}
        for line, crow in [(cfg.line1, c1), (cfg.line2, c2)]:
            ctl_model = delta_from_rows(crow[:cfg.n_ctl_model])
            src_t = torch.from_numpy(ctl_model).float().to(device)
            pert_id_b = torch.tensor(vocab.encode(p), dtype=torch.long, device=device).reshape(1, 1)
            xhat = ode_predict(vf, gene_ids, src_t, pert_id_b, cfg.batch_size, cfg.ode_steps,
                               cfg.noise_type, cfg.poisson_alpha, cfg.poisson_target_sum, device)
            xhat = xhat.cpu().numpy()
            deltas_hat[line] = xhat.mean(0) - ctl_model.mean(0)

        d1_real = real_delta(cfg.line1, p, c1)
        d2_real = real_delta(cfg.line2, p, c2)

        # 排除靶基因自身
        keep = np.ones(len(modeled), dtype=bool)
        if p in modeled:
            keep[modeled.index(p)] = False

        r_hat_sp = spearmanr(deltas_hat[cfg.line1][keep], deltas_hat[cfg.line2][keep]).statistic
        r_real_sp = spearmanr(d1_real[keep], d2_real[keep]).statistic
        r_hat_pe = pearsonr(deltas_hat[cfg.line1][keep], deltas_hat[cfg.line2][keep]).statistic
        r_real_pe = pearsonr(d1_real[keep], d2_real[keep]).statistic

        rows_out.append({'pert': p, 'r_hat_spearman': r_hat_sp, 'r_real_spearman': r_real_sp,
                         'r_hat_pearson': r_hat_pe, 'r_real_pearson': r_real_pe})
        all_deltas[p] = {f'{cfg.line1}_hat': deltas_hat[cfg.line1],
                         f'{cfg.line2}_hat': deltas_hat[cfg.line2],
                         f'{cfg.line1}_real': d1_real, f'{cfg.line2}_real': d2_real}
        log(f'{p}: r_hat_sp={r_hat_sp:.3f} r_real_sp={r_real_sp:.3f} '
            f'| r_hat_pe={r_hat_pe:.3f} r_real_pe={r_real_pe:.3f}')

    df = pd.DataFrame(rows_out)
    csv_path = os.path.join(cfg.out_dir, 'line_cond_test.csv')
    df.to_csv(csv_path, index=False)
    np.savez(os.path.join(cfg.out_dir, 'deltas.npz'), **{k.replace('.', '_'): v for k, v in all_deltas.items()},
             modeled=np.array(modeled))

    log('\n===== summary =====')
    for col in ['r_hat_spearman', 'r_real_spearman', 'r_hat_pearson', 'r_real_pearson']:
        log(f'mean {col}: {df[col].mean():.4f}  (median {df[col].median():.4f})')
    log(f'model cross-line spearman {df["r_hat_spearman"].mean():.4f} vs real {df["r_real_spearman"].mean():.4f}')
    verdict = ('模型侧显著更高 -> 条件混合假设成立' if df['r_hat_spearman'].mean() > df['r_real_spearman'].mean() + 0.15
               else '两侧量级接近 -> 假设被削弱')
    log(f'verdict: {verdict}')
    log(f'csv -> {csv_path}')
    log('TEST_DONE')


if __name__ == '__main__':
    main()
