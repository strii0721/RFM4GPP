"""2026-10-04 全轴 PCA 拟合 v4（GPU 分块 randomized SVD，用户定案：1024 维、不筛 HVG）。

v1（scipy 稀疏精确 XᵀX）灾难：协方差近稠密、CSR 构建 50k 行 >11min。
v2（scipy 全矩阵 densify）灾难：CSR.toarray 500k 行单核 17min+ 未完。
v3（z-score 版）重建灾难：2749 个近零方差基因 sd clamp 到 1e-6 → z 爆炸
  （|z| 达 5.6e6）→ real 重建 R²=-31、采样块自重建仅 0.21。
v4 定案：**均值中心化 log1p 空间 PCA（不除 sd）**——scanpy 不 scale 的自然
  形态，高方差基因主导主成分（等效软 HVG 加权），无除法爆炸。
  采样 N_BLOCKS 个连续块 → 每块稀疏直读 GPU → 减 μ → 手动两遍 randomized
  SVD（Y=Σ(A−μ)Ω，Q=qr(Y)，B=Σ Qᵀ(A−μ)，小矩阵 SVD → W）。

投影：x_pc = (x − μ) @ W；回投影 x̂ = μ + x_pc @ Wᵀ。std 仅作参考存档。

用法：
  python -m src.script.fit_pca --cache_dir <dir> --out <npz> \
      [--n_blocks 4 --block_rows 25000 --q 1024]
"""
import argparse
import os
import time

import numpy as np
import torch


def _stem_of(cache_dir: str) -> str:
    for fn in sorted(os.listdir(cache_dir)):
        if fn.endswith('.h5ad.data.npy'):
            return fn[:-len('.data.npy')]
    raise FileNotFoundError(f'no processed_*.h5ad.data.npy under {cache_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_blocks', type=int, default=4)
    ap.add_argument('--block_rows', type=int, default=25000)
    ap.add_argument('--q', type=int, default=1024, help='主成分数')
    ap.add_argument('--oversample', type=int, default=128)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    dev = torch.device(args.device)
    stem = _stem_of(args.cache_dir)
    base = os.path.join(args.cache_dir, stem)
    data = np.load(base + '.data.npy', mmap_mode='r')
    indices = np.load(base + '.indices.npy', mmap_mode='r')
    indptr = np.load(base + '.indptr.npy', mmap_mode='r')
    n_rows = len(indptr) - 1

    import h5py
    with h5py.File(base + '.meta.h5ad', 'r') as f:
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f['var']['_index'][:]]
    n_cols = len(names)

    span = n_rows // args.n_blocks
    starts = [i * span for i in range(args.n_blocks)]
    r = args.q + args.oversample
    print(f'[fit_pca v4] rows={n_rows} cols={n_cols} blocks={args.n_blocks}×'
          f'{args.block_rows} q={args.q} r={r} (mean-centered, no scale)',
          flush=True)

    def load_block_gpu(st):
        """读块 → GPU dense (block_rows, n_cols) fp32。"""
        s = int(indptr[st])
        e = int(indptr[st + args.block_rows])
        ip = torch.from_numpy(np.asarray(indptr[st:st + args.block_rows + 1]) - s)
        idx = torch.from_numpy(np.asarray(indices[s:e]))
        val = torch.from_numpy(np.asarray(data[s:e]))
        sp = torch.sparse_csr_tensor(ip, idx, val,
                                     size=(args.block_rows, n_cols))
        return sp.to(dev).to_dense()

    def iter_blocks():
        for st in starts:
            yield load_block_gpu(st)

    # pass0：μ 与 σ（σ 仅存档参考）流式统计
    t0 = time.time()
    sum_x = torch.zeros(n_cols, dtype=torch.float64, device=dev)
    sum_x2 = torch.zeros(n_cols, dtype=torch.float64, device=dev)
    for dense in iter_blocks():
        d = dense.double()
        sum_x += d.sum(0)
        sum_x2 += d.pow(2).sum(0)
    n_s = args.n_blocks * args.block_rows
    mu = sum_x / n_s
    sd = torch.sqrt(torch.clamp(sum_x2 / n_s - mu ** 2, min=0.0))
    print(f'[fit_pca v4] pass0 stats {time.time() - t0:.0f}s', flush=True)

    # pass1：Y = Σ (A−μ) Ω
    Omega = torch.randn(n_cols, r, device=dev)
    Y = torch.zeros(n_s, r, device=dev)
    mut = mu.float()
    for bi, dense in enumerate(iter_blocks()):
        off = bi * args.block_rows
        Y[off:off + args.block_rows] = (dense - mut) @ Omega
        del dense
    print(f'[fit_pca v4] pass1 Y {time.time() - t0:.0f}s', flush=True)

    Q, _ = torch.linalg.qr(Y)
    del Y
    QQ = Q.t().contiguous()

    # pass2：B = Σ Qᵀ (A−μ)
    B = torch.zeros(r, n_cols, device=dev)
    for bi, dense in enumerate(iter_blocks()):
        off = bi * args.block_rows
        B += QQ[:, off:off + args.block_rows] @ (dense - mut)
        del dense
    print(f'[fit_pca v4] pass2 B {time.time() - t0:.0f}s', flush=True)

    U2, S2, Vt = torch.linalg.svd(B, full_matrices=False)
    W = Vt.t().cpu().numpy()[:, :args.q]
    evals = (S2.cpu().numpy() ** 2 / max(n_s - 1, 1))[:args.q]
    np.savez_compressed(args.out,
                        components=W.astype(np.float32),
                        mean=mu.cpu().numpy().astype(np.float32),
                        std=sd.cpu().numpy().astype(np.float32),
                        eigenvalues=evals.astype(np.float32),
                        gene_names=np.array(names, dtype=object))
    # 中心化空间解释方差比（分母 = 总中心化方差）
    tot_var = float((sum_x2 / n_s - mu ** 2).sum().item())
    expl = evals.sum() / max(tot_var, 1e-12)
    print(f'[fit_pca v4] saved {args.out}  top-{args.q} 解释方差比 {expl:.3f}  '
          f'总耗时 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
