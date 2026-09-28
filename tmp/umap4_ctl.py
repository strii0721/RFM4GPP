#!/usr/bin/env python3
"""4 数据集对照组 UMAP：official controls / iscr12g / xatlas / replogle_bad
label 直接标图上（用户偏好）；附 PCA50 空间质心距离表（定量判断谁离官方最近）。

2026-09-15 v2：xatlas(93.8GB)/replogle(15.8GB) 绕开 anndata backed（实测 obs 装载
病态膨胀 457GB RSS 且 >1.5h），改用 h5py 直读 codes 选行 + 连续段读 X。
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad
import h5py
import pandas as pd
from scipy import sparse

if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True

ROOT = '/ssd1/ict2/Projects/vcc-2026/resources/datasets'
OUT = '/ssd1/ict2/Projects/vcc-2026/output'
os.makedirs(OUT, exist_ok=True)

CAP = 1500
SEED = 42
rng = np.random.default_rng(SEED)
N_GENES = 18533


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def read_ctl_h5py(path, line_col, src_tag):
    """h5py 直读：codes 选对照组 → 每 (系) 组 cap → 连续段读 X 行。"""
    log(f'h5py-read {path.split("/")[-1]} ...')
    f = h5py.File(path, 'r')
    tg = f['obs/target_gene']
    tg_cats = [x.decode() for x in tg['categories'][:]]
    nt = tg_cats.index('non-targeting')
    m = (tg['codes'][:] == nt)
    g = f['obs/' + line_col]
    g_cats = [x.decode() for x in g['categories'][:]]
    g_codes = g['codes'][:]
    log(f'  ctl={int(m.sum())}, groups={g_cats}')
    picks = []
    for ci, cname in enumerate(g_cats):
        gi = np.nonzero(m & (g_codes == ci))[0]
        n = min(CAP, len(gi))
        picks.append((cname, np.sort(rng.choice(gi, size=n, replace=False))))
    sel = np.sort(np.concatenate([p[1] for p in picks]))
    # per-row label
    row_label = np.empty(len(sel), dtype=object)
    for cname, gi_sel in picks:
        for r in gi_sel:
            row_label[np.searchsorted(sel, r)] = f'{cname}_{src_tag}'
    log(f'  selected {len(sel)} rows, reading X ...')
    indptr = f['X/indptr'][:]
    data = f['X/data']
    indices = f['X/indices']
    X = np.zeros((len(sel), N_GENES), dtype=np.float32)
    B = 500
    for b in range(0, len(sel), B):
        rows = sel[b:b + B]
        s = int(indptr[rows[0]])
        e = int(indptr[rows[-1] + 1])
        d = data[s:e]
        idx = indices[s:e]
        cnt = indptr[rows + 1] - indptr[rows]
        ends = np.cumsum(cnt)
        starts = ends - cnt
        for j in range(len(rows)):
            X[b + j, idx[starts[j]:ends[j]]] = d[starts[j]:ends[j]]
        log(f'  rows {min(b + B, len(sel))}/{len(sel)} done')
    f.close()
    return ad.AnnData(X=sparse.csr_matrix(X),
                      obs=pd.DataFrame({'label': row_label}))


def read_ctl_anndata(path, line_col, src_tag):
    log(f'anndata-read {path.split("/")[-1]} ...')
    a = sc.read_h5ad(path, backed='r')
    tg = a.obs['target_gene'].astype(str).to_numpy()
    m = (tg == 'non-targeting')
    ln = a.obs[line_col].astype(str).to_numpy()
    log(f'  ctl={int(m.sum())}, groups={np.unique(ln[m]).tolist()}')
    keep_mask = np.zeros(a.n_obs, dtype=bool)
    for g in np.unique(ln[m]):
        gi = np.nonzero(m & (ln == g))[0]
        n = min(CAP, len(gi))
        sel = np.sort(rng.choice(gi, size=n, replace=False))
        keep_mask[sel] = True
    sub = a[keep_mask].to_memory()
    if src_tag == 'context':
        sub.obs['label'] = src_tag + '_' + sub.obs[line_col].astype(str).to_numpy()
    else:
        sub.obs['label'] = sub.obs[line_col].astype(str).to_numpy() + '_' + src_tag
    sub.obs = sub.obs[['label']].copy()
    log(f'  kept {sub.shape[0]} cells')
    return sub


SOURCES = [
    (f'{ROOT}/controls/context_A.h5ad', 'context', 'context', read_ctl_anndata),
    (f'{ROOT}/controls/context_B.h5ad', 'context', 'context', read_ctl_anndata),
    (f'{ROOT}/controls/context_C.h5ad', 'context', 'context', read_ctl_anndata),
    (f'{ROOT}/train/iscr12g/iscr12g_aligned.h5ad', 'cell_line', 'iscr12g', read_ctl_anndata),
    (f'{ROOT}/train/xatlas/x_atlas_aligned.h5ad', 'context', 'xatlas', read_ctl_h5py),
    (f'{ROOT}/train/replogle_bad/replogle_bad_aligned.h5ad', 'context', 'replogle', read_ctl_h5py),
]

parts = []
for path, line_col, src_tag, reader in SOURCES:
    parts.append(reader(path, line_col, src_tag))

log('concatenating ...')
full = ad.concat(parts, join='outer', index_unique=None)
full.obs['label'] = full.obs['label'].astype('category')
log(f'total {full.shape[0]} cells x {full.shape[1]} genes')
log(full.obs['label'].value_counts().to_string())

log('normalize/log1p/HVG/PCA/neighbors/UMAP ...')
sc.pp.normalize_total(full, target_sum=1e4)
sc.pp.log1p(full)
sc.pp.highly_variable_genes(full, n_top_genes=3000)
full = full[:, full.var['highly_variable']].copy()
sc.pp.pca(full, n_comps=50)
sc.pp.neighbors(full, n_pcs=50, n_neighbors=15)
sc.tl.umap(full, min_dist=0.3)
log('umap done, saving ...')

full.write_h5ad(os.path.join(OUT, 'umap4_ctl.h5ad'))

fig, ax = plt.subplots(figsize=(16, 11))
sc.pl.umap(full, color='label', legend_loc='on data', legend_fontsize=6,
           title=f'CONTROL-only UMAP: official(3) + iscr12g + xatlas + replogle (n={full.n_obs})',
           show=False, ax=ax)
fig.savefig(os.path.join(OUT, 'umap4_ctl.png'), dpi=150, bbox_inches='tight')
log('PNG saved to ' + os.path.join(OUT, 'umap4_ctl.png'))

# ---- PCA50 空间质心距离（欧氏）：train 组 vs 官方 3 context ----
X = full.obsm['X_pca']
labels = full.obs['label'].to_numpy()
official = [f'context_{c}' for c in 'ABC']
rows = []
for lab in sorted(set(labels)):
    if lab in official:
        continue
    xi = X[labels == lab]
    ds = [np.linalg.norm(xi.mean(0) - X[labels == oc].mean(0)) for oc in official]
    rows.append([lab] + [f'{d:.3f}' for d in ds] + [official[int(np.argmin(ds))]])

import csv
csv_path = os.path.join(OUT, 'umap4_pca_centroid_dist.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group', 'dist_context_A', 'dist_context_B', 'dist_context_C', 'nearest'])
    w.writerows(rows)
log('\nPCA50 centroid distance to official contexts (smaller = closer):')
log('group | dA | dB | dC | nearest')
for r in sorted(rows, key=lambda r: sum(float(x) for x in r[1:4])):
    log(' | '.join(r))
log('CSV saved to ' + csv_path)
log('UMAP_DONE')
