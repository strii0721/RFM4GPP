#!/usr/bin/env python
"""构建残差目标冻结张量（rbar_c / rbar_p / gbar / res_<line>），2026-09-26 用户定案范式。

口径（逐字定案）：
  logFC r[c,p] = log2((x̄_pert + 1) / (x̄_ctrl + 1))，x̄ 为逐细胞 CPM(1e6) 后组内
  算术均值；Res[c,p] = r − r̄_c − r̄_p + ḡ（r̄_c 系内均值、r̄_p 跨系均值、ḡ 全局均值）。
  单系扰动的 Res = ḡ − r̄_c 为常量（退化，用户接受）。

输入输出全部由 configs/universal.yaml 派生（2026-09-28 用户定案），flag 仅覆盖：
  - 语料：common.train_set_paths 全列表按序读入（多文件沿 (line, pert) 累积合并）
  - 缓存基因序：common.train_cache_dir + flow.n_top_genes 派生的 processed_*.meta.h5ad
  - 输出：common.frozen_tensors_dir

用法（远程项目根）:
  .venv/bin/python -m src.script.build_tensors          # 全默认（YAML 派生）
  .venv/bin/python -m src.script.build_tensors --adata_path <x.h5ad> --cache_meta <m.h5ad>

产物（2026-09-30 mrn19843 口径：语料轴=缓存列=19,843 基因）：
  combos.csv          (line, pert, row) —— row = res_<line>.npy 行号
  res_<line>.npy      float32 (n_perts_line, 19843)
  rbar_p.npy + rbar_p_perts.csv     跨系扰动主效应
  rbar_c.npy + rbar_c_lines.csv     系主效应
  gbar.npy           全局均值
  genes_cache.csv    缓存列基因名（对齐序）

运行日志自动落 logs/tensors_<ts>/build.log（2026-09-26 目录规范，脚本自建）。
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from src.utils.config_utils import CommonConfig, ConfigUtils, FlowConfig
from src.utils.utils import derive_pert_columns


def _self_log(task: str, log_base: str) -> None:
    """自建日志目录 <log_base>/<task>_<ts>/build.log 并重定向 stdout/stderr（2026-09-26 目录规范）。"""
    log_dir = os.path.join(log_base, f'{task}_{time.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(log_dir, exist_ok=True)
    f = open(os.path.join(log_dir, 'build.log'), 'a', buffering=1)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    sys.stdout = f
    sys.stderr = f


def cpm_col_mean(X, rows):
    """逐细胞 CPM(1e6) 后组内算术均值 = 列加权均值（精确口径）。"""
    sub = X[rows]
    tot = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
    w = 1e6 / np.maximum(tot, 1.0)
    return np.asarray(sub.T @ w).ravel() / len(w)


def _accumulate_file(path, acc_num, acc_cnt, genes0, cfg):
    """单文件累加 (line, pert) -> CPM 列均值加权分子/计数（跨文件合并用）。

    cfg.perturb_direction：扰动方向白名单。非空时按 exo_perturb_subtype 剔除
    不在列表内的细胞（keep 掩码与文件行号对齐，2026-09-28 定案：与缓存构建
    同口径）；cfg.obs_col_candidates/ctrl_sentinels 供 derive_pert_columns 匹配。
    """
    a = ad.read_h5ad(path, backed='r')
    obs = derive_pert_columns(a.obs, cfg.obs_col_candidates, cfg.ctrl_sentinels)
    if genes0 is not None:
        assert list(a.var_names) == genes0, f'var 轴不一致: {path}'
    n_genes = a.n_vars
    keep = np.ones(a.n_obs, dtype=bool)
    if cfg.perturb_direction and 'exo_perturb_subtype' in a.obs:
        keep = a.obs['exo_perturb_subtype'].astype(str).isin(cfg.perturb_direction).to_numpy()
    tg_all = obs['target_gene'].astype(str).to_numpy()
    ctx_all = obs['context'].astype(str).to_numpy()
    lines = sorted(set(ctx_all[keep]))
    print(f'[res] file {path}: {a.shape}, kept {keep.sum()}/{a.n_obs} by '
          f'perturb_direction={cfg.perturb_direction}, lines={lines}', flush=True)
    for line in lines:
        lm = (ctx_all == line) & keep
        cells = np.nonzero(lm)[0]
        lo, hi = cells.min(), cells.max() + 1
        # 2026-10-03 分块（用户定案二）：全文件块 int64 读入 + 转换中间
        # 实测 ~100GB/worker（14 并发 OOM）；按 20 万行分块，峰值 ~20GB。
        chunk = 200_000
        for clo in range(lo, hi, chunk):
            chi = min(clo + chunk, hi)
            block = a[clo:chi]
            Xb = block.X.tocsr().astype(np.float32)
            tg_b = tg_all[clo:chi]
            k_b = keep[clo:chi]
            del block
            if not k_b.all():
                Xb = Xb[k_b]
                tg_b = tg_b[k_b]
            # 2026-10-03 批量算法：扰动指示矩阵 P（每扰动一行 0/1）→
            # P @ (Xb·w) 一次算出全部扰动的加权列和（O(nnz) 单遍）。
            rowsum = np.asarray(Xb.sum(axis=1)).ravel().astype(np.float64)
            w = 1e6 / np.maximum(rowsum, 1.0)
            tg_codes, tg_uniq = pd.factorize(tg_b)
            n_p = len(tg_uniq)
            order = np.argsort(tg_codes, kind='stable')
            tg_sorted = tg_codes[order]
            bounds = np.nonzero(tg_sorted[1:] != tg_sorted[:-1])[0] + 1
            P_ptr = np.concatenate([[0], bounds, [len(tg_b)]])
            P = sparse.csr_matrix((np.ones(len(tg_b)), order, P_ptr),
                                  shape=(n_p, len(tg_b)))
            Xw = Xb.multiply(w[:, None]).tocsr()
            PM = np.asarray((P @ Xw).todense())     # (n_p, n_genes) 加权列和
            cnts = np.bincount(tg_codes, minlength=n_p)
            for j, tg in enumerate(tg_uniq):
                num = acc_num.get((line, str(tg)), np.zeros(n_genes))
                cnt = acc_cnt.get((line, str(tg)), 0)
                acc_num[(line, str(tg))] = num + PM[j]
                acc_cnt[(line, str(tg))] = cnt + int(cnts[j])
            del Xb, Xw, PM, P
    a.file.close()
    return list(a.var_names) if genes0 is None else genes0


def _accumulate_file_worker(args):
    """worker（2026-10-03 文件级并行）：单文件独立累积局部 dict。"""
    path, cfg = args
    acc_num: dict = {}
    acc_cnt: dict = {}
    genes = _accumulate_file(path, acc_num, acc_cnt, None, cfg)
    return acc_num, acc_cnt, genes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adata_path', default=None,
                    help='覆盖：单语料文件（默认取 common.train_set_paths 全列表）')
    ap.add_argument('--cache_meta', default=None,
                    help='覆盖：缓存 meta h5ad（默认由 train_cache_dir 派生）')
    ap.add_argument('--out_dir', default=None,
                    help='覆盖：输出目录（默认 common.frozen_tensors_dir）')
    args = ap.parse_args()

    common = ConfigUtils.load(CommonConfig, use_cli=False)
    fcfg = ConfigUtils.load(FlowConfig, use_cli=False)
    out_dir = args.out_dir or common.frozen_tensors_dir

    stem = '+'.join(os.path.splitext(os.path.basename(str(p)))[0]
                    for p in sorted(common.train_set_paths))
    if len(stem) > 80:  # 2026-10-03：与 data.py/_corpus_stem 同口径 hash 短名
        import hashlib
        stem = 'h' + hashlib.sha1(stem.encode()).hexdigest()[:24]
    pool_stem = (os.path.splitext(os.path.basename(str(fcfg.train_pool_path)))[0]
                 if fcfg.train_pool_path else 'all')
    cache_meta = args.cache_meta or os.path.join(
        common.train_cache_dir,
        f'processed_n{fcfg.n_top_genes}_{stem}_{pool_stem}.h5ad.meta.h5ad')

    _self_log('tensors', common.log_base_dir)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    paths = [args.adata_path] if args.adata_path else list(common.train_set_paths)
    assert paths, 'train_set_paths 为空且未给 --adata_path'

    # ---- 多文件累积：(line, pert) -> CPM 加权均值分子/计数
    # 2026-10-03 文件级并行（用户定案）：每文件独立累积局部 dict，主进程同键相加。
    import multiprocessing as mp
    n_workers = min(getattr(common, 'cache_workers', 1), 8, len(paths))
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_accumulate_file_worker, [(p, fcfg) for p in paths])
    acc_num, acc_cnt = {}, {}
    genes0 = results[0][2]
    n_genes0 = len(genes0)
    for rn, rc, rg in results:
        assert rg == genes0, 'worker 返回 var 轴不一致'
        for k, v in rn.items():
            num = acc_num.get(k, np.zeros(n_genes0))
            cnt = acc_cnt.get(k, 0)
            acc_num[k] = num + v
            acc_cnt[k] = cnt + rc[k]
    genes_full = genes0
    lines = sorted({k[0] for k in acc_num})
    print(f'[res] accumulated {len(acc_num)} (line, pert) combos, lines={lines}', flush=True)

    # 缓存列对齐（缓存已建时按 meta 对齐；未建时全轴恒等——n_top_genes==语料 var 轴）
    if os.path.exists(cache_meta):
        meta = ad.read_h5ad(cache_meta)
        cache_genes = list(meta.var_names)
    else:
        cache_genes = list(genes_full)
        print(f'[res] cache meta 未建（{cache_meta}），按全轴恒等对齐（n_top_genes==语料 var 轴）',
              flush=True)
    pos_full = {g: i for i, g in enumerate(genes_full)}
    keep_pos = np.array([pos_full[g] for g in cache_genes], dtype=np.int64)
    assert len(keep_pos) == len(cache_genes), 'cache meta genes not found in corpus var'
    print(f'[res] cache genes {len(cache_genes)}, aligned to corpus var', flush=True)

    # ---- r[c,p]（全轴 19,843，每组合一行）
    r_rows, combos = [], []
    ctl_mean = {line: acc_num[(line, 'non-targeting')] / acc_cnt[(line, 'non-targeting')]
                for line in lines}
    for (line, p) in sorted(acc_num):
        if p == 'non-targeting':
            continue
        pm = acc_num[(line, p)] / acc_cnt[(line, p)]
        cm = ctl_mean[line]
        r = np.log2((pm + 1.0) / (cm + 1.0)).astype(np.float32)
        r_rows.append(r)
        combos.append((line, p))
    r_mat = np.vstack(r_rows)                      # (n_combo, 19843)
    combos_arr = np.array(combos)
    print(f'[res] r computed {r_mat.shape} in {time.time()-t0:.0f}s', flush=True)

    # ---- 主效应与全局均值（每组合一行等权）
    n = len(combos)
    rbar_c = np.vstack([r_mat[combos_arr[:, 0] == L].mean(axis=0) for L in lines])
    rbar_p = {p: r_mat[combos_arr[:, 1] == p].mean(axis=0) for p in sorted(set(combos_arr[:, 1]))}
    gbar = r_mat.mean(axis=0)
    rbar_p_arr = np.vstack([rbar_p[p] for p in sorted(rbar_p)])
    rbar_p_perts = sorted(rbar_p)
    # Res[c,p] = r − r̄_c − r̄_p + ḡ
    res = r_mat - rbar_c[[lines.index(c) for c in combos_arr[:, 0]]] \
          - np.vstack([rbar_p[p] for p in combos_arr[:, 1]]) + gbar[None, :]
    print(f'[res] Res stats: mean={res.mean():.3e} std={res.std():.4f} '
          f'max_abs={np.abs(res).max():.3f}', flush=True)
    # 单系扰动退化验证：Res == ḡ − r̄_c（逐基因）
    n_lines_per = {p: int((combos_arr[:, 1] == p).sum()) for p in rbar_p_perts}
    single = [p for p, k in n_lines_per.items() if k == 1]
    if single:
        p0 = single[0]
        i0 = np.nonzero(combos_arr[:, 1] == p0)[0][0]
        exp = gbar - rbar_c[lines.index(combos_arr[i0, 0])]
        print(f'[res] single-line check {p0}: max|Res−(ḡ−r̄_c)|='
              f'{np.abs(res[i0] - exp).max():.2e}（应≈0）', flush=True)

    # ---- 对齐缓存列后落盘
    res = res[:, keep_pos].astype(np.float32)
    rbar_p_arr = rbar_p_arr[:, keep_pos].astype(np.float32)
    rbar_c = rbar_c[:, keep_pos].astype(np.float32)
    gbar = gbar[keep_pos].astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(combos_arr, columns=['line', 'pert']).to_csv(
        os.path.join(out_dir, 'combos.csv'), index=False)
    for L in lines:
        idx = np.nonzero(combos_arr[:, 0] == L)[0]
        np.save(os.path.join(out_dir, f'res_{L}.npy'), res[idx])
    np.save(os.path.join(out_dir, 'rbar_p.npy'), rbar_p_arr)
    pd.DataFrame({'pert': rbar_p_perts}).to_csv(
        os.path.join(out_dir, 'rbar_p_perts.csv'), index=False)
    np.save(os.path.join(out_dir, 'rbar_c.npy'), rbar_c)
    pd.DataFrame({'line': lines}).to_csv(os.path.join(out_dir, 'rbar_c_lines.csv'),
                                         index=False)
    np.save(os.path.join(out_dir, 'gbar.npy'), gbar)
    pd.DataFrame({'gene': cache_genes}).to_csv(os.path.join(out_dir, 'genes_cache.csv'),
                                               index=False)
    print(f'[res] saved to {out_dir}: {len(combos)} combos, '
          f'{len(rbar_p_perts)} perts, {time.time()-t0:.0f}s total', flush=True)


if __name__ == '__main__':
    main()
