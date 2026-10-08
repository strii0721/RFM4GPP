"""训练可学习编解码器（2026-10-05 用户定案：A 方案，全量 25.4M 细胞，纯 MSE）。

用法（8 卡，远程项目根）：
  torchrun --nproc_per_node=8 -m src.script.train_autoencoder --epochs 3 --batch 512
单卡冒烟：
  .venv/bin/python -m src.script.train_autoencoder --epochs 1 --batch 512 --max-steps 50

产物（rank0 落盘 output/autoencoder_<数据集后缀>/，文件名带训练时间戳；2026-10-07 定案）：
  encoder_<ts>.pt / decoder_<ts>.pt（每 epoch 覆盖同名，kill 后即最后完成 epoch 的权重）
  train.log（loss + 固定 eval 细胞集上的重建保真）
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from scipy import sparse
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.autoencoder import AEEncoder, AEDecoder  # noqa: E402
from src.data_process.data import _corpus_stem, _row_slice_csr  # noqa: E402
from src.utils.config_utils import ConfigUtils, FlowConfig  # noqa: E402


def _eval_recon(enc, dec, x, device):
    """固定细胞集上的重建保真（centered per-gene corr 中位 + MSE）。x 预转常驻 GPU。"""
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        z, skips = enc(x)
        r = dec(z, skips)
    xn = x.float().cpu().numpy()
    rn = r.float().cpu().numpy()
    xc = xn - xn.mean(0, keepdims=True)
    rc = rn - rn.mean(0, keepdims=True)
    num = (xc * rc).sum(0)
    den = np.sqrt((xc ** 2).sum(0) * (rc ** 2).sum(0)) + 1e-12
    corrs = num / den
    mse = float(((xn - rn) ** 2).mean())
    return mse, float(np.median(corrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)   # warm-start 口径（2026-10-06 用户定案：3 epoch）
    ap.add_argument('--batch', type=int, default=512)   # 每卡 batch（2026-10-06 用户定：1024→512）
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=8192)
    ap.add_argument('--latent', type=int, default=2048)   # 2026-10-07 加深版：latent 2048（h1=8192, h2=4096）
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--max-steps', type=int, default=0, help='>0 时限制总步数（冒烟用）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cfg = ConfigUtils.load(FlowConfig, use_cli=False)
    # 缓存路径按 data.py/generate_submission 同款规则派生（换轴即用，不硬编码 stem）
    stem = _corpus_stem(cfg)
    pool_stem = (os.path.splitext(os.path.basename(str(cfg.train_pool_path)))[0]
                 if cfg.train_pool_path else 'all')
    cache = os.path.join(cfg.train_cache_dir,
                         f'processed_n{cfg.n_top_genes}_{stem}_{pool_stem}.h5ad')

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(rank)
    dev = torch.device(f'cuda:{rank}')

    # ---- 数据：侧车零拷贝挂载 ----
    d = np.load(cache + '.data.npy', mmap_mode='r')
    idxs = np.load(cache + '.indices.npy', mmap_mode='r')
    ptr = np.load(cache + '.indptr.npy', mmap_mode='r')
    N, n_genes = ptr.shape[0] - 1, cfg.n_top_genes
    X = sparse.csr_matrix((N, n_genes), dtype=d.dtype)
    X.data, X.indices, X.indptr = d, idxs, ptr
    X.has_sorted_indices = True
    if rank == 0:
        print(f'cache: N={N} genes={n_genes} nnz_avg={len(d)/max(N,1):.0f}', flush=True)

    # ---- 模型 ----
    enc = AEEncoder(n_genes, args.hidden, args.latent).to(dev)
    dec = AEDecoder(n_genes, args.hidden, args.latent).to(dev)
    model = torch.nn.ModuleDict({'encoder': enc, 'decoder': dec})
    if world > 1:
        model = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = N // world // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    # ---- 输出目录（2026-10-07 用户定案：autoencoder_<数据集后缀>，后缀取自缓存目录名
    # cache_<后缀> 去前缀；固定 ckpt 目录，文件名带训练时间戳）
    _cache_stem = os.path.basename(cfg.train_cache_dir.rstrip('/'))
    if _cache_stem.startswith('cache_'):
        _cache_stem = _cache_stem[len('cache_'):]
    ckpt_dir = os.path.join(cfg.output_base_dir, f'autoencoder_{_cache_stem}')
    run_ts = os.environ.get('SCDFM_RUN_TS') or time.strftime('%Y-%m-%d_%H-%M')
    if rank == 0:
        # 日志自落盘（2026-10-07 用户定案：与其他阶段同口径，任务名 pretrain_autoencoder）
        _log_dir = os.path.join(cfg.log_base_dir, f'pretrain_autoencoder_{run_ts}')
        os.makedirs(_log_dir, exist_ok=True)
        _log_f = open(os.path.join(_log_dir, 'train.log'), 'a', buffering=1)
        os.dup2(_log_f.fileno(), 1)
        os.dup2(_log_f.fileno(), 2)
        sys.stdout = _log_f
        sys.stderr = _log_f
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f'ckpt_dir={ckpt_dir} run_ts={run_ts} epochs={args.epochs} '
              f'batch={args.batch} steps/rank/epoch={steps_per_epoch}', flush=True)

    rng = np.random.default_rng(args.seed)
    eval_idx = np.sort(rng.choice(N, 8192, replace=False))
    _esub = _row_slice_csr(X, eval_idx)
    eval_x = torch.sparse_csr_tensor(
        torch.from_numpy(_esub.indptr).to(torch.int64),
        torch.from_numpy(_esub.indices).to(torch.int64),
        torch.from_numpy(_esub.data).to(torch.float32),
        size=(len(eval_idx), n_genes),
    ).to(dev).to_dense().bfloat16()
    del _esub

    def get_enc_dec():
        m = model.module if world > 1 else model
        return m['encoder'], m['decoder']

    step = 0
    enc_m, dec_m = get_enc_dec()   # DDP 下取 unwrapped 子模块（循环外一次）
    # 顺序流式（2026-10-06 终态，用户定案）：不 shuffle、不预读入内存——
    # 每卡自己的连续 1/world 段顺序 mmap 扫（顺序 IO 吃满 NFS 带宽，页用后即弃）。
    chunk = (N + world - 1) // world
    my_lo, my_hi = rank * chunk, min(N, (rank + 1) * chunk)
    import queue
    import threading
    for epoch in range(args.epochs):
        # 预取线程（2026-10-06，4 worker 用户定案）：后台提前把下一批 nnz 段读进
        # 内存（掩蔽 NFS 偶发慢读与 licong 的 IO 争抢）；主循环从队列取现成段。
        n_w = 4   # 2026-10-06 实测定案：4 最优（8 过饱和——NFS 上限 2.2GB/s、4 已吃 68%）
        q = queue.Queue(maxsize=4 * n_w)
        def _pf(w):
            for s in range(my_lo + w * args.batch, my_hi, n_w * args.batch):
                e = min(s + args.batch, my_hi)
                n0, n1 = int(X.indptr[s]), int(X.indptr[e])
                q.put((e - s,
                       np.ascontiguousarray(X.indptr[s:e + 1] - n0),
                       np.ascontiguousarray(X.indices[n0:n1].astype(np.int64)),
                       np.ascontiguousarray(X.data[n0:n1])))
            q.put(None)
        for w in range(n_w):
            threading.Thread(target=_pf, args=(w,), daemon=True).start()
        done_cnt = 0
        while done_cnt < n_w:
            item = q.get()
            if item is None:
                done_cnt += 1
                continue
            n_rows, sub_ptr, sub_idx, sub_data = item
            # 连续行快路径（2026-10-06 顺序流式红利）：nnz 段连续零循环
            x = torch.sparse_csr_tensor(
                torch.from_numpy(sub_ptr),
                torch.from_numpy(sub_idx),
                torch.from_numpy(sub_data),
                size=(n_rows, n_genes),
            ).to(dev).to_dense().bfloat16()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z, skips = enc_m(x)
                r = dec_m(z, skips)
                loss = torch.nn.functional.mse_loss(r, x)
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪（2026-10-07 修复 loss 爆炸：防单步大梯度打飞权重）
            torch.nn.utils.clip_grad_norm_(list(enc_m.parameters()) + list(dec_m.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            if rank == 0 and step % args.eval_every == 0:
                e, dd = get_enc_dec()
                mse, corr = _eval_recon(e, dd, eval_x, dev)
                print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] [e{epoch} s{step}] '
                      f'eval: MSE={mse:.4f} per-gene corr med={corr:.4f}',
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        if rank == 0:
            e, dd = get_enc_dec()
            torch.save(e.state_dict(), os.path.join(ckpt_dir, f'encoder_{run_ts}.pt'))
            torch.save(dd.state_dict(), os.path.join(ckpt_dir, f'decoder_{run_ts}.pt'))
            print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] [e{epoch}] '
                  f'saved encoder_{run_ts}.pt / decoder_{run_ts}.pt', flush=True)

    if rank == 0:
        e, dd = get_enc_dec()
        torch.save(e.state_dict(), os.path.join(ckpt_dir, f'encoder_{run_ts}.pt'))
        torch.save(dd.state_dict(), os.path.join(ckpt_dir, f'decoder_{run_ts}.pt'))
        print(f'DONE -> {ckpt_dir}', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
