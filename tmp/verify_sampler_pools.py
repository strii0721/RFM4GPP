#!/usr/bin/env python3
"""TrainSampler 池构建新旧实现等价验证（2026-09-26 向量化后一次性核对）。

旧实现逐 (pert, line) 全列字符串比较；新实现每系一次 stable argsort 切分。
验证：前 N_CHECK 个扰动（按排序后 _perturbation_covariates 顺序）的
_tgt_pool 键集/索引数组/_eligible_lines 完全一致；并对全体扰动跑一遍新实现
输出池计数（秒级），供与旧运行日志的 12,701 对照。

用法（远程项目根）: .venv/bin/python -u src/script/verify_sampler_pools.py
"""
import sys
import time

import numpy as np
import anndata as ad

sys.path.insert(0, '.')
sys.path.insert(0, '/home/ict2/Projects/scDFM')

CACHE = '/home/ict2/Projects/scDFM/tmp/vcc/processed_n11919_replogle_k562_jurkat_hepg2_all.h5ad'
MIN_TGT = 20  # 训练配置 min_tgt_cells=20
N_CHECK = 200

a = ad.read_h5ad(CACHE, backed='r')
obs = a.obs
# 与 process_data vcc 分支一致：Drug1/Drug2 由 condition 派生（缓存 obs 无这两列）
obs['Drug1'] = obs['condition'].str.split('+').str[0]
obs['Drug2'] = obs['condition'].str.split('+').str[-1]
obs['perturbation_covariates'] = obs[['Drug1', 'Drug2']].apply(lambda x: '+'.join(x), axis=1)
lines = obs['context'].astype(str).to_numpy()
pc = obs['perturbation_covariates'].astype(str).to_numpy()
perts_all = np.array([p for p in np.unique(pc) if p != 'control+control'])
perts_all.sort()
line_names = np.unique(lines)
print(f'cells={a.n_obs} perts={len(perts_all)} lines={list(line_names)}', flush=True)


def old_build(perts):
    ctl = {L: np.nonzero(np.logical_and(lines == L, pc == 'control+control'))[0]
           for L in line_names}
    tgt, elig = {}, {}
    for pert in list(perts):
        el = []
        for L in line_names:
            idx = np.nonzero(np.logical_and(lines == L, pc == pert))[0]
            if len(idx) >= MIN_TGT and len(ctl[L]) > 0:
                tgt[(pert, L)] = idx
                el.append(L)
        elig[pert] = el
    return tgt, elig


def new_build(perts):
    ctl = {L: np.nonzero(np.logical_and(lines == L, pc == 'control+control'))[0]
           for L in line_names}
    tgt, elig_map = {}, {}
    for L in line_names:
        mask_L = lines == L
        base = np.nonzero(mask_L)[0]
        pc_L = pc[mask_L]
        order = np.argsort(pc_L, kind='stable')
        spc = pc_L[order]
        bounds = np.nonzero(spc[1:] != spc[:-1])[0] + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [spc.size]))
        ctl_ok = len(ctl[L]) > 0
        for s, e in zip(starts, ends):
            pert = spc[s]
            if pert == 'control+control':
                continue
            if (e - s) >= MIN_TGT and ctl_ok:
                tgt[(pert, L)] = base[order[s:e]]
                elig_map.setdefault(pert, []).append(L)
    elig = {p: elig_map.get(p, []) for p in list(perts)}
    return tgt, elig


# 子集等价核对
sub = perts_all[:N_CHECK]
t0 = time.time()
old_tgt, old_elig = old_build(sub)
t_old = time.time() - t0
t0 = time.time()
new_tgt, new_elig = new_build(sub)
t_new = time.time() - t0
print(f'subset check: old={t_old:.1f}s new={t_new:.1f}s '
      f'({len(old_tgt)} vs {len(new_tgt)} pools)', flush=True)

new_tgt_sub = {k: v for k, v in new_tgt.items() if k[0] in set(sub)}
only_old = set(old_tgt) - set(new_tgt_sub)
only_new = set(new_tgt_sub) - set(old_tgt)
if only_old or only_new:
    print(f'DIAG only_old={len(only_old)} e.g. {list(only_old)[:5]}', flush=True)
    print(f'DIAG only_new={len(only_new)} e.g. {list(only_new)[:5]}', flush=True)
assert not only_old and not only_new, 'tgt pool key sets differ'
for k in old_tgt:
    assert np.array_equal(old_tgt[k], new_tgt_sub[k]), f'index mismatch {k}'
for p in sub:
    assert old_elig[p] == new_elig[p], f'elig mismatch {p}'
print(f'SUBSET EQUIVALENT: {len(old_tgt)} pools identical over {N_CHECK} perts', flush=True)

# 全体新实现（秒级）计数
t0 = time.time()
full_tgt, full_elig = new_build(perts_all)
n_elig = sum(1 for p in perts_all if full_elig[p])
print(f'full new build: {time.time()-t0:.1f}s, {len(full_tgt)} pools, '
      f'{n_elig} eligible perts (expect ~12,701 pools)', flush=True)
