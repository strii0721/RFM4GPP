#!/usr/bin/env python3
"""训练侧 DEG 索引 —— 双口径（2026-09-24 用户定案）。

主口径（= 我们之前的方法 / VCC 评测器同款，de_replogle_deg_union 定稿）：
  每 (line, target) 桶 = 扰动 min(100, 实际) 细胞 vs 同 line 对照 min(4000, 实际)
  （seed 42；<20 细胞跳过）→ counts→CPM(1e6)→log1p（MWU 秩不变，raw counts 直接
  等价）→ pdex MWU（two-sided+continuity，numba_mwu = cell-eval2 官方 DE 后端）
  → ref 算术均值 CPM>5 门控（reference-only，门控外 p 置 1）→ BH per 桶 0.05
  → 剔自身靶基因。
  产物：deg_sets_vcc.csv（跨 line 并集 padj<=0.05：pert,gene,dir,padj,n_lines,lines,conflict）

次口径（PerturbQA A.2 参照）：无表达量门控、全扰动细胞、BH 后 padj<0.01、
  剔自身靶。产物：deg_sets_pqa.csv。

并行：pdex.pdex 的 group 循环串行（实测单核）——multiprocessing fork（COW 共享
csr）+ 每 worker 串行 mwu（numba 线程池置 1 防过订阅）。

产物（out_dir）：deg_long_<line>.csv（p<0.05 未门控行）、deg_sets_vcc.csv、
  deg_sets_pqa.csv、summary.csv（每扰动 n_cells/两口径 DEG 数/is_panel）

运行日志自动落 logs/deg_index_<ts>/build.log（2026-09-26 目录规范，脚本自建）。

用法（远程项目根）:
  .venv/bin/python -u src/script/build_deg_index.py \
      --adata_path /home/ict2/Projects/vcc-2026/resources/datasets/replogle/replogle_k562_jurkat_hepg2.h5ad \
      --out_dir output/deg_index --n_ctrl 4000 --n_workers 64
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import anndata as ad


def _self_log(task: str) -> None:
    """自建日志目录 logs/<task>_<ts>/build.log 并重定向 stdout/stderr（2026-09-26 目录规范）。"""
    log_dir = os.path.join('logs', f'{task}_{time.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(log_dir, exist_ok=True)
    f = open(os.path.join(log_dir, 'build.log'), 'a', buffering=1)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    sys.stdout = f
    sys.stderr = f

_SHARED_X = None
_REF_X = None
_REF_MEAN = None


def _one_group(rows):
    import pdex as _p
    _p.set_numba_threadpool(1)
    x = _SHARED_X[rows]
    pv = np.asarray(_p.mwu(x, _REF_X).pvalue).clip(0, 1)
    x_mean = np.asarray(x.mean(axis=0)).ravel()
    lfc = np.asarray(_p.log2_fold_change(x_mean, _REF_MEAN, epsilon=1.0)).ravel()
    return pv, lfc


def _t(tag, t0):
    print(f'    {tag}: {time.time()-t0:.1f}s', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adata_path', required=True)
    ap.add_argument('--out_dir', default='output/deg_index')
    ap.add_argument('--n_ctrl', type=int, default=4000)
    ap.add_argument('--n_pert_cap', type=int, default=100)
    ap.add_argument('--min_cells', type=int, default=20)
    ap.add_argument('--gate_cpm', type=float, default=5.0)
    ap.add_argument('--n_workers', type=int, default=64)
    args = ap.parse_args()
    _self_log('deg_index')
    os.makedirs(args.out_dir, exist_ok=True)

    t_all = time.time()
    a = ad.read_h5ad(args.adata_path, backed='r')
    print(f'[deg] corpus {a.shape}, workers={args.n_workers}', flush=True)
    tg_all = a.obs['target_gene'].astype(str).to_numpy()
    line_all = a.obs['cell_line_name'].astype(str).to_numpy()
    genes = np.asarray(a.var_names)
    lines = sorted(set(line_all))
    _t(f'obs ready, lines={lines}', t_all)

    panel = set()
    panel_path = '/home/ict2/Projects/vcc-2026/resources/datasets/replogle/pert_counts.csv'
    if os.path.exists(panel_path):
        panel = set(pd.read_csv(panel_path)['target_gene'].astype(str))

    rng = np.random.default_rng(42)
    deg_vcc, deg_pqa, summaries = [], [], []
    ctx = mp.get_context('fork')
    for line in lines:
        t0 = time.time()
        lm = line_all == line
        cells = np.nonzero(lm)[0]
        tg_line = tg_all[cells]
        ntc_idx = cells[tg_line == 'non-targeting']
        if len(ntc_idx) > args.n_ctrl:
            ntc_idx = np.sort(rng.choice(ntc_idx, size=args.n_ctrl, replace=False))
        pert_cells = cells[tg_line != 'non-targeting']
        keep_idx = np.sort(np.concatenate([pert_cells, ntc_idx]))
        # backed h5 散点索引只建视图，tocsr() 时才逐行物化（实测病态慢）——
        # 先按连续范围读块（h5 快路径），再内存内选行
        lo, hi = keep_idx.min(), keep_idx.max() + 1
        block = a[lo:hi]
        X_block = block.X.tocsr().astype(np.float32)
        X = X_block[keep_idx - lo]
        _t(f'[{line}] block [{lo},{hi}) {X.shape[0]} rows', t0)
        tg_sub = tg_all[keep_idx]
        ref_pos = np.searchsorted(keep_idx, ntc_idx)
        ref_X = X[ref_pos]
        # ref 算术均值 CPM（精确：每细胞 CPM 后逐细胞平均 = 列加权均值）
        ref_tot = np.asarray(ref_X.sum(axis=1)).ravel().astype(np.float64)
        w = 1e6 / np.maximum(ref_tot, 1.0)
        ref_mean_cpm = np.asarray(ref_X.T @ w).ravel() / len(w)
        # ref count 均值（log2fc 用）
        ref_mean = np.asarray(ref_X.mean(axis=0)).ravel()
        _t('ref means + gate', t0)

        n_cells_all = pd.Series(tg_sub[tg_sub != 'non-targeting']).value_counts()
        groups = [(g, np.nonzero(tg_sub == g)[0])
                  for g in sorted(set(tg_sub) - {'non-targeting'})]
        # 扰动细胞封顶 + <min_cells 跳过（评测器口径；跳过组不进 csv）
        rows_list, names = [], []
        for g, rows in groups:
            if len(rows) < args.min_cells:
                continue
            if len(rows) > args.n_pert_cap:
                rows = np.sort(rng.choice(rows, size=args.n_pert_cap, replace=False))
            names.append(g)
            rows_list.append(rows)
        _t(f'groups {len(names)} (cap {args.n_pert_cap}, min {args.min_cells})', t0)

        global _SHARED_X, _REF_X, _REF_MEAN
        _SHARED_X, _REF_X, _REF_MEAN = X, ref_X, ref_mean
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
            futs = [pool.submit(_one_group, rows) for rows in rows_list]
            for i, f in enumerate(futs):
                f.result()
                if (i + 1) % 500 == 0 or i + 1 == len(futs):
                    print(f'[{line}] {i+1}/{len(futs)} groups done, '
                          f'{time.time()-t0:.0f}s', flush=True)
            results = [f.result() for f in futs]

        pv_mat = np.vstack([r[0] for r in results])
        lfc_mat = np.vstack([r[1] for r in results])
        _t('mwu all groups', t0)

        # ---- VCC 口径：门控外 p=1 → BH → padj<=0.05 → 剔自身靶
        gate_ok = ref_mean_cpm > args.gate_cpm
        pv_gated = np.where(gate_ok[None, :], pv_mat, 1.0)
        padj_vcc = pdex.false_discovery_control(pv_gated, axis=1, method='bh')
        # ---- PQA 口径：无门控 → BH → padj<0.01
        padj_pqa = pdex.false_discovery_control(pv_mat, axis=1, method='bh')

        keep = pv_mat < 0.05
        ri, ci = np.nonzero(keep)
        df = pd.DataFrame({
            'pert': [names[i] for i in ri],
            'gene': genes[ci],
            'pvalue': pv_mat[keep],
            'padj_vcc': padj_vcc[keep],
            'padj_pqa': padj_pqa[keep],
            'log2fc': lfc_mat[keep],
            'ref_cpm': ref_mean_cpm[ci],
            'line': line,
        })
        df.to_csv(os.path.join(args.out_dir, f'deg_long_{line}.csv'), index=False)

        n_genes = pv_mat.shape[1]
        pert_arr = np.array(names)
        self_mask = pert_arr[:, None] == genes[None, :]
        is_de_vcc = (padj_vcc <= 0.05) & ~self_mask
        is_de_pqa = (padj_pqa < 0.01) & ~self_mask
        dr = np.sign(lfc_mat).astype(int)
        for mask, store, tag in [(is_de_vcc, deg_vcc, 'vcc'), (is_de_pqa, deg_pqa, 'pqa')]:
            for i in np.nonzero(mask.any(axis=1))[0]:
                gi = np.nonzero(mask[i])[0]
                store.append(pd.DataFrame({
                    'pert': pert_arr[i], 'gene': genes[gi], 'dir': dr[i, gi],
                    'padj': padj_vcc[i, gi] if tag == 'vcc' else padj_pqa[i, gi],
                    'line': line,
                }))
        s = pd.DataFrame({
            'pert': pert_arr,
            'n_cells': [int(n_cells_all.get(g, 0)) for g in pert_arr],
            'n_deg_vcc': is_de_vcc.sum(axis=1),
            'n_deg_pqa': is_de_pqa.sum(axis=1),
            'is_panel': [g in panel for g in pert_arr],
        })
        s['line'] = line
        summaries.append(s)
        print(f'[{line}] vcc {int(is_de_vcc.sum())} / pqa {int(is_de_pqa.sum())} DEG rows, '
              f'{time.time()-t0:.0f}s total', flush=True)
        del _SHARED_X, _REF_X, _REF_MEAN, X, ref_X, block, X_block

    for store, name, alpha in [(deg_vcc, 'deg_sets_vcc', 'padj<=0.05'),
                               (deg_pqa, 'deg_sets_pqa', 'padj<0.01')]:
        if not store:
            continue
        d = pd.concat(store, ignore_index=True).sort_values('padj')
        g = d.groupby(['pert', 'gene'], as_index=False).agg(
            padj=('padj', 'min'),
            dir=('dir', lambda x: x.iloc[0]),
            n_lines=('line', 'nunique'),
            lines=('line', lambda x: ','.join(sorted(set(x)))),
            dir_conflict=('dir', lambda x: int(len(set(x)) > 1)),
        )
        g.to_csv(os.path.join(args.out_dir, f'{name}.csv'), index=False)
        print(f'[deg] {name}: {len(g)} DEG pairs across {g["pert"].nunique()} perturbations '
              f'({alpha})', flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(os.path.join(args.out_dir, 'summary.csv'),
                                                   index=False)
    print(f'[deg] all done {time.time()-t_all:.0f}s', flush=True)


if __name__ == '__main__':
    import pdex  # noqa: F401
    main()
