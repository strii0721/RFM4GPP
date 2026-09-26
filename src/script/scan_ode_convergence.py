#!/usr/bin/env python3
"""ODE 步数收敛性扫描（训练完成后、正式推理前；2026-09-26 用户定案）。

对冻结模型，在同一批对照细胞 + 同一初始噪声下，扫 N ∈ {12,25,50,100,200}，
比较 Euler 数值解 x(1)（= Reŝ；r̂ = r̄_p − ḡ + Reŝ 只差常数，收敛性等价），
找出"再翻倍步数 x(1) 变化足够小"的最小 N，作为正式 300 基因推理的步数依据。

任务模式（一进程一 (gene, N)，由 scan_ode_dispatch.py 分发）:
  .venv/bin/python -u src/script/scan_ode_convergence.py \
      --checkpoint_path output/train_<ts>/iteration_N/checkpoint.pt \
      --real_path output/benchmark_<ts>/real.h5ad \
      --gene ACIN1 --n_steps 100 --gpu 3 \
      --out_dir output/scan_<ts>
  → 写 {out_dir}/{gene}_n{N}.npy（float32 (n_cells, 11371)）

报告模式:
  .venv/bin/python -u src/script/scan_ode_convergence.py \
      --report 1 --genes ACIN1,ARID5B,BRD4 --n_list 12,25,50,100,200 \
      --out_dir output/scan_<ts>
  → 逐基因相邻 N 的 mean/max |Δx| 表 + 按阈值推荐 N

关键设计：同 gene 跨 N 的对照细胞（seed 派生）与初始噪声（stable_seed 派生）
完全一致，Δx 只反映 Euler 截断误差，不含抽样噪声。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tyro

from config.config_flow import FlowConfig
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.script.benchmark_line_holdout import _norm_log1p
from src.script.generate_submission import (
    artifact_paths,
    ode_predict,
    select_modeled_genes,
    stable_seed,
)


class ScanConfig(FlowConfig):
    checkpoint_path: str = ''
    real_path: str = ''        # 复用 benchmark 轮次的 real.h5ad（含 4000 NTC 对照）
    out_dir: str = ''
    gene: str = ''             # 任务模式：单基因
    n_steps: int = 100
    n_cells: int = 400
    seed: int = 42
    gpu: int = 0
    report: bool = False
    genes: str = ''            # 报告模式：逗号分隔基因
    n_list: str = '12,25,50,100,200'
    threshold: float = 0.005   # 推荐判据：Δx 均值 < 此值（log2FC 单位）
    top_infer_genes: int = 11919  # 建模基因数（与 BenchConfig 同，select_modeled_genes 需要）


def _load_model(cfg: ScanConfig, device):
    cache, mask_path, vocab_path = artifact_paths(cfg)
    vocab = GeneVocab.from_file(vocab_path)
    modeled = select_modeled_genes(cache, cfg.panel_path, cfg.top_infer_genes, vocab,
                                   pool_path=cfg.train_pool_path)
    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    vf = instantiate_model(cfg.model_type, ntoken=cfg.ntoken, d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path)
    ckpt = torch.load(cfg.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    return vf.to(device).eval(), gene_ids, vocab, modeled


def run_task(cfg: ScanConfig) -> None:
    assert cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path), 'checkpoint_path required'
    assert cfg.real_path and os.path.exists(cfg.real_path), 'real_path required'
    assert cfg.gene, '--gene required in task mode'
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f'{cfg.gene}_n{cfg.n_steps}.npy')
    if os.path.exists(out_path):
        print(f'skip existing {out_path}', flush=True)
        return

    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    vf, gene_ids, vocab, modeled = _load_model(cfg, device)
    print(f'model loaded: {len(modeled)} modeled genes, n_steps={cfg.n_steps}', flush=True)

    # 对照细胞：与 build_pred 同口径（全量 NTC 里 seed 随机抽 n_cells），跨 N 一致
    real = sc.read_h5ad(cfg.real_path)
    tg = real.obs['target_gene'].astype(str).to_numpy()
    ctl_idx = np.nonzero(tg == 'non-targeting')[0]
    rng = np.random.default_rng(cfg.seed)
    src_idx = np.sort(rng.choice(len(ctl_idx), size=min(cfg.n_cells, len(ctl_idx)), replace=False))
    ctl_raw = real.X[ctl_idx[src_idx]].tocsr()
    ctl_norm = _norm_log1p(ctl_raw)
    gene_axis_pos = {g: i for i, g in enumerate(real.var_names)}
    modeled_pos = np.array([gene_axis_pos[g] for g in modeled], dtype=np.int64)
    src_modeled = torch.from_numpy(ctl_norm[:, modeled_pos].toarray()).float().to(device)
    print(f'control cells: {len(src_idx)} x {len(modeled)} modeled genes', flush=True)

    # 初始噪声：同基因跨 N 完全一致（stable_seed 派生）
    torch.manual_seed(stable_seed(cfg.heldout_line, cfg.gene, cfg.seed + 13))
    noise = torch.randn(len(src_idx), len(modeled), device=device)

    pert_id_b = torch.tensor([vocab.encode(cfg.gene)], dtype=torch.long,
                             device=device).repeat(1, 1)
    x1 = ode_predict(
        vf, gene_ids, src_modeled, pert_id_b, cfg.batch_size, cfg.n_steps,
        'Gaussian', getattr(cfg, 'poisson_alpha', 0.8),
        getattr(cfg, 'poisson_target_sum', 1e4), device,
        clamp_output=False,  # 残差空间可负
        noise=noise,
    ).cpu().numpy()
    np.save(out_path, x1.astype(np.float32))
    print(f'saved {out_path} ({x1.shape}, |x1| mean={np.abs(x1).mean():.4f})', flush=True)


def run_report(cfg: ScanConfig) -> None:
    genes = [g.strip() for g in cfg.genes.split(',') if g.strip()]
    n_list = [int(n) for n in cfg.n_list.split(',')]
    n_list_sorted = sorted(n_list)
    assert len(genes) and len(n_list) >= 2, '--genes and --n_list (>=2) required'
    rows = []
    for g in genes:
        xs = {}
        for n in n_list:
            p = os.path.join(cfg.out_dir, f'{g}_n{n}.npy')
            assert os.path.exists(p), f'missing {p}'
            xs[n] = np.load(p)
        for i in range(1, len(n_list_sorted)):
            n_prev, n_cur = n_list_sorted[i - 1], n_list_sorted[i]
            d = np.abs(xs[n_cur] - xs[n_prev])
            rows.append({'gene': g, 'N': n_prev, 'N_next': n_cur,
                         'mean|d|': float(d.mean()), 'max|d|': float(d.max()),
                         'p99|d|': float(np.percentile(d, 99))})
    df = pd.DataFrame(rows)
    print('\n===== ODE 步数收敛性（|Δx|，log2FC 单位；Δ = 与下一个 N 比较）=====')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    # 推荐：每基因找第一个 mean|d| < threshold 的 (N -> 下一 N) 对，取 N_next 作为该基因收敛点
    recs = {}
    for g in genes:
        sub = df[df.gene == g].sort_values('N')
        pick = None
        for _, r in sub.iterrows():
            if r['mean|d|'] < cfg.threshold:
                pick = int(r['N_next'])
                break
        recs[g] = pick
    print('\n推荐（mean|Δx| < {:.4f} 的最小 N_next）：'.format(cfg.threshold))
    for g, n in recs.items():
        print(f'  {g}: {n if n else "未收敛（需扩展 n_list）"}')
    picks = [n for n in recs.values() if n]
    if picks:
        print(f'→ 建议正式推理步数：{max(picks)}（取各基因收敛点的最大值）')


def main() -> None:
    cfg = tyro.cli(ScanConfig, description=__doc__)
    if cfg.report:
        run_report(cfg)
    else:
        run_task(cfg)


if __name__ == '__main__':
    main()
