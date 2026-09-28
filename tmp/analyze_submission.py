#!/usr/bin/env python3
"""Analyze the merged submission h5ad (in-memory; remote box has ~1TB RAM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 2026-09-28: tmp/ 下减一层

import numpy as np
import pandas as pd

import anndata as ad

path = sys.argv[1]
controls_dir = sys.argv[2] if len(sys.argv) > 2 else '/home/ict2/Projects/vcc-2026/resources/datasets/controls'

a = ad.read_h5ad(path)
X = a.X.tocsr()
obs = a.obs
gene_names = list(a.var_names)

print('== overall ==', flush=True)
print('shape', a.shape, 'obs', list(a.obs.columns), flush=True)
for ctx in ['A', 'B', 'C']:
    m = (obs['context'] == ctx).to_numpy()
    depths = np.asarray(X[m].sum(axis=1)).ravel()
    print(f'ctx {ctx}: {m.sum()} cells, nnz/cell {X[m].nnz/m.sum():.0f}, '
          f'depth mean {depths.mean():.0f} med {np.median(depths):.0f} max {depths.max():.0f}', flush=True)

# control per-gene means (one pass over each control file)
print('== target-gene KD (predicted mean vs control mean) ==', flush=True)
pert_list = sorted(obs['target_gene'].unique())
ctl_means = {}
for ctx in ['A', 'B', 'C']:
    c = ad.read_h5ad(f'{controls_dir}/context_{ctx}.h5ad')
    cX = c.X.tocsr()
    ctl_means[ctx] = np.asarray(cX.sum(axis=0)).ravel() / c.n_obs
    del c

# row blocks: each (ctx, pert) is 400 consecutive rows (concat preserved pair order)
rows = []
labels = obs[['context', 'target_gene']].to_numpy()
boundaries = np.flatnonzero(np.any(labels[1:] != labels[:-1], axis=1)) + 1
blocks = [(s, e) for s, e in zip(np.r_[0, boundaries], np.r_[boundaries, len(labels)])]
assert len(blocks) == 900, f'expected 900 blocks, got {len(blocks)}'
assert all(e - s == 400 for s, e in blocks), 'block size != 400'

kd_rows = []
for (s, e) in blocks:
    ctx, pert = labels[s]
    gi = gene_names.index(pert)
    pred_mean = np.asarray(X[s:e, gi].toarray()).ravel().mean()
    ctl_mean = ctl_means[ctx][gi]
    kd_rows.append((ctx, pert, pred_mean / max(ctl_mean, 1e-9), pred_mean, ctl_mean))
kd = pd.DataFrame(kd_rows, columns=['ctx', 'pert', 'ratio', 'pred_mean', 'ctl_mean'])
for ctx in ['A', 'B', 'C']:
    r = kd[kd.ctx == ctx]['ratio']
    print(f'ctx {ctx}: ratio mean {r.mean():.3f} med {r.median():.3f}, '
          f'frac<1 {(r < 1).mean():.2%}, frac<0.5 {(r < 0.5).mean():.2%}, frac<0.2 {(r < 0.2).mean():.2%}', flush=True)
kd.to_csv(path.replace('.h5ad', '_kd_analysis.csv'), index=False)

# discriminability: mean pairwise cosine distance of pert mean profiles per context
print('== perturbation discriminability ==', flush=True)
rng = np.random.default_rng(0)
sample_genes = rng.choice(len(gene_names), 3000, replace=False)
for ctx in ['A', 'B', 'C']:
    means = []
    for (s, e) in blocks[:120]:  # first 120 blocks are ctx A
        c, p = labels[s]
        if c != ctx:
            continue
        v = np.asarray(X[s:e, sample_genes].mean(axis=0)).ravel()
        n = np.linalg.norm(v)
        means.append(v / n if n > 0 else v)
    M = np.stack(means)
    S = M @ M.T
    d = 1 - S[np.triu_indices(len(means), 1)]
    print(f'ctx {ctx}: {len(means)} perts, mean pairwise cosine dist {d.mean():.4f}', flush=True)

print('done', flush=True)
