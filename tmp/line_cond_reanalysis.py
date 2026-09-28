#!/usr/bin/env python3
"""line_cond_test 复分析：仅响应基因口径（2026-09-15）。

对每个扰动，按两系真实效应幅度排序取 top-K 响应基因，
在该基因子集上重算模型侧 vs 真实侧的跨系 Spearman/Pearson 相关。
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

NPZ = '/ssd1/ict2/Projects/vcc-2026/output/line_cond_test/deltas.npz'
OUT = '/ssd1/ict2/Projects/vcc-2026/output/line_cond_test/line_cond_responsive.csv'
TOPK = 100

d = np.load(NPZ, allow_pickle=True)
perts = [k for k in d.files if k not in ('modeled',)]
print(f'perturbations: {len(perts)}')

rows = []
for p in perts:
    arr = d[p].item()
    d1h, d2h = arr['HEK293T_hat'], arr['K562_hat']
    d1r, d2r = arr['HEK293T_real'], arr['K562_real']
    # 响应基因 = 两系真实 |Δ| 取 max 排 top-K
    resp = np.maximum(np.abs(d1r), np.abs(d2r))
    sel = np.argsort(resp)[::-1][:TOPK]
    r_hat_sp = spearmanr(d1h[sel], d2h[sel]).statistic
    r_real_sp = spearmanr(d1r[sel], d2r[sel]).statistic
    r_hat_pe = pearsonr(d1h[sel], d2h[sel]).statistic
    r_real_pe = pearsonr(d1r[sel], d2r[sel]).statistic
    rows.append({'pert': p, 'r_hat_spearman': r_hat_sp, 'r_real_spearman': r_real_sp,
                 'r_hat_pearson': r_hat_pe, 'r_real_pearson': r_real_pe,
                 'n_resp_gene': len(sel)})
    print(f'{p}: r_hat_sp={r_hat_sp:.3f} r_real_sp={r_real_sp:.3f} | '
          f'r_hat_pe={r_hat_pe:.3f} r_real_pe={r_real_pe:.3f}')

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print('\n===== summary (top-%d responsive genes) =====' % TOPK)
for col in ['r_hat_spearman', 'r_real_spearman', 'r_hat_pearson', 'r_real_pearson']:
    print(f'mean {col}: {df[col].mean():.4f}  (median {df[col].median():.4f})')
print(f'\nmodel cross-line spearman {df["r_hat_spearman"].mean():.4f} '
      f'vs real {df["r_real_spearman"].mean():.4f}')
print(f'csv -> {OUT}')
print('REANALYSIS_DONE')
