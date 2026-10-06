"""将 VCC controls 三文件（ABC，轴 18,533）对齐到训练轴 19,547：
按 genes_cache.csv 模板重排列 + 缺失基因补零列，输出到 mrnh19547 目录。
原始文件不动（2026-10-06 用户定案：副本对齐）。"""
import os
import sys

import anndata as ad
ad.settings.allow_write_nullable_strings = True  # ABC obs 含 pd.StringArray（2026-10-06）
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTROLS_DIR = '/home/ict2/Projects/vcc-2026/resources/datasets/controls'
OUT_DIR = '/home/ict2/Projects/vcc-2026/resources/datasets/mrnh19547'
GENES_CSV = '/home/ict2/Projects/scDFM/output/frozen_tensors_mrnh19547/genes_cache.csv'


def main():
    template = pd.read_csv(GENES_CSV)['gene'].astype(str).tolist()
    assert len(template) == 19547, f'template axis {len(template)} != 19547'
    for tag in ('A', 'B', 'C'):
        src = os.path.join(CONTROLS_DIR, f'context_{tag}.h5ad')
        dst = os.path.join(OUT_DIR, f'context_{tag}_n19547.h5ad')
        a = ad.read_h5ad(src)
        names = a.var_names.astype(str).tolist()
        pos = {g: i for i, g in enumerate(names)}
        missing = [g for g in template if g not in pos]
        print(f'context_{tag}: {a.shape} -> axis {len(template)}, missing {len(missing)} genes')
        # 对齐（19547 轴重排 + 缺基因补零）；X 以 csr 稀疏存储（推理侧假设，2026-10-06）
        x = np.asarray(a.X.todense() if hasattr(a.X, 'todense') else a.X, dtype=np.float32)
        out = np.zeros((a.n_obs, len(template)), dtype=np.float32)
        for j, g in enumerate(template):
            if g in pos:
                out[:, j] = x[:, pos[g]]
        from scipy import sparse as sp_sparse
        new = ad.AnnData(X=sp_sparse.csr_matrix(out), obs=a.obs.copy(),
                         var=pd.DataFrame(index=template))
        os.makedirs(OUT_DIR, exist_ok=True)
        new.write_h5ad(dst)
        # 回读校验
        chk = ad.read_h5ad(dst, backed='r')
        assert chk.shape == (a.n_obs, 19547), chk.shape
        assert list(chk.var_names.astype(str)[:3]) == template[:3]
        print(f'  wrote {dst} {chk.shape} OK')


if __name__ == '__main__':
    main()
