"""提交前轴裁剪（2026-09-30 用户定案）：模型输出 19,843 轴 → 官方提交轴 18,533。

用法:
  .venv/bin/python tmp/crop_to_vcc_axis.py --in pred.h5ad \
      --gene_csv .../mrn19843/gene_names_vcc.csv --out pred_18533.h5ad

行为: var 重排到 gene_csv 轴序；目标轴缺失基因补 0 列（18,533 中仅官方有的
486 个）、训练轴独有基因丢弃（1,310 个）；obs 原样保留。
"""
import argparse
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--gene_csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    target = pd.read_csv(args.gene_csv)['gene_name'].astype(str).tolist()
    a = ad.read_h5ad(args.inp)
    pos = {g: i for i, g in enumerate(a.var_names.astype(str))}
    idx = np.array([pos.get(g, -1) for g in target], dtype=np.int64)
    valid = idx >= 0
    Xv = a.X[:, idx[valid]]
    if sp.issparse(Xv):
        out = sp.lil_matrix((a.n_obs, len(target)), dtype=Xv.dtype)
        out[:, valid] = Xv.tocsr()
        X = out.tocsr()
    else:
        X = np.zeros((a.n_obs, len(target)), dtype=Xv.dtype)
        X[:, valid] = Xv
    b = ad.AnnData(X=X, obs=a.obs.copy(),
                   var=pd.DataFrame(index=target))
    b.write_h5ad(args.out)
    print(f'crop {a.shape} -> {b.shape} '
          f'(missing zero-pad={int((~valid).sum())}, dropped={int(a.n_vars - valid.sum())})',
          flush=True)


if __name__ == '__main__':
    sys.exit(main())
