import h5py
import numpy as np
from collections import Counter

PANEL = set()
with open('/ssd1/ict2/Projects/vcc-2026/resources/datasets/controls/pert_counts.csv') as f:
    next(f)
    for line in f:
        PANEL.add(line.strip())
print('panel genes:', len(PANEL), flush=True)


def scan(path, linecol, tag):
    f = h5py.File(path, 'r')
    tg = f['obs/target_gene']
    cats = [x.decode() for x in tg['categories'][:]]
    codes = tg['codes'][:]
    g = f['obs/' + linecol]
    gcats = [x.decode() for x in g['categories'][:]]
    gcodes = g['codes'][:]
    print(f'--- {tag} ({path.split("/")[-1]}) ---', flush=True)
    rows = []
    for ci, cname in enumerate(gcats):
        m = gcodes == ci
        n = int(m.sum())
        if n == 0:
            continue
        sub = codes[m]
        cc = Counter(cats[c] for c in sub)
        n_ctl = cc.get('non-targeting', 0)
        pert = {g_: c for g_, c in cc.items() if g_ != 'non-targeting'}
        panel_genes = [g_ for g_ in pert if g_ in PANEL]
        cells_panel = sum(pert[g_] for g_ in panel_genes)
        cnts = sorted((pert[g_] for g_ in panel_genes), reverse=True)
        ge20 = sum(1 for c in cnts if c >= 20)
        ge100 = sum(1 for c in cnts if c >= 100)
        ge400 = sum(1 for c in cnts if c >= 400)
        med = np.median(cnts) if cnts else 0
        print(f'{cname}: total={n} ctl={n_ctl} panel_cells={cells_panel} '
              f'panel_genes={len(panel_genes)} (>=20:{ge20} >=100:{ge100} >=400:{ge400}) '
              f'median_cells_per_gene={med:.0f}', flush=True)
        rows.append((tag, cname, n, n_ctl, cells_panel, len(panel_genes), ge20, ge100, ge400, med))
    f.close()
    return rows


all_rows = []
all_rows += scan('/ssd1/ict2/Projects/vcc-2026/resources/datasets/train/xatlas/x_atlas_aligned.h5ad',
                 'context', 'xatlas')
all_rows += scan('/ssd1/ict2/Projects/vcc-2026/resources/datasets/train/replogle_bad/replogle_bad_aligned.h5ad',
                 'context', 'replogle')
all_rows += scan('/ssd1/ict2/Projects/vcc-2026/resources/datasets/train/iscr12g/iscr12g_aligned.h5ad',
                 'cell_line', 'iscr12g')
all_rows.sort(key=lambda r: -r[4])  # by panel cells
print('\n===== ranked by panel perturbation cells =====', flush=True)
for r in all_rows:
    print(f'{r[0]:8s} {r[1]:14s} panel_cells={r[4]:>8d} genes={r[5]:>3d} '
          f'(>=100:{r[7]:>3d}) ctl={r[3]:>7d} total={r[2]:>8d} med={r[9]:.0f}', flush=True)
print('SCAN_DONE', flush=True)
