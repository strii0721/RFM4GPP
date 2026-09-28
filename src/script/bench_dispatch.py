#!/usr/bin/env python3
"""守护分发进程（2026-09-24 起支持跨机双实例）：
每 poll_s(10) 秒轮询本机 GPU 显存，空闲（可用 > free_mb=50G）的卡立刻领下一个
panel 基因的推理任务。每基因一个独立进程：跑完立即写
predparts/pred_shard_disp_{host}_{task}_g000.h5ad 并退出（释放显存），下个轮询
周期该卡若仍空闲则继续领任务；全部基因完成后（跨机验证完整）自动 eval_only
出最终六指标（scores.csv 落在 out_dir）。

跨机协调（.36/.49 各跑一个实例，共享盘同一 out_dir）：
  - 单实例锁按 host 分离（.dispatch.pid.<host>），两机各持一个锁
  - 基因认领 = predparts/.claims/<gene> O_EXCL 原子创建（NFS 原子），跨机互斥；
    启动时清理本机死 pid 的遗留 claim 与 >24h 的陈旧 claim
  - 任务编号/日志/part 文件名含 host 前缀，跨机不冲突
  - 最终 eval 由 O_EXCL 认领（.eval.claim）且轮询等全部基因完成才执行，
    避免一机先跑完时另一机仍在飞 → eval 收口不完整

与旧分片机制的区别：单基因粒度调度 + 显存门控——别人占用某卡（可用 <50G）时
该卡自动被跳过，别人腾出后自动恢复；进程级失败自动重试（--max_retry）。

断电/重启安全：启动时扫描 predparts 里全部基因级 part（含旧分片命名），已完成
基因自动跳过；dispatch_map_<host>.txt 记录 disp 任务号→基因映射（编号续接）。

用法（在项目根目录，.36 与 .49 各跑一份）:
  .venv/bin/python -m src.script.bench_dispatch \
      --checkpoint_path output/train_2026-09-22_11-20/iteration_21000/checkpoint.pt \
      --out_dir output/benchmark_2026-09-26_14-00 --gpus 8 --ode_steps 100
"""
import argparse
import glob
import os
import random
import socket
import subprocess
import sys
import time

import anndata as ad

HOST = socket.gethostname().split('.')[0]


def free_mem_mib() -> dict[int, int]:
    """{gpu_idx: free_mib}，解析 nvidia-smi 的 total/used（MiB）。"""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.strip().splitlines():
        idx, tot, used = (int(x.strip()) for x in line.split(','))
        res[idx] = tot - used
    return res


def load_done(out_dir: str) -> set[str]:
    """扫描 predparts 全部基因级 part（含旧分片命名），解析已完成基因名集合。"""
    done = set()
    for p in glob.glob(os.path.join(out_dir, 'predparts', '*_g*.h5ad')):
        try:
            a = ad.read_h5ad(p)
            done.add(str(a.obs['target_gene'].iloc[0]))
        except Exception as e:
            print(f'[warn] 跳过坏 part {os.path.basename(p)}: {e}', flush=True)
    return done


def claim_dir(out_dir: str) -> str:
    d = os.path.join(out_dir, 'predparts', '.claims')
    os.makedirs(d, exist_ok=True)
    return d


def claimed_genes(out_dir: str) -> set[str]:
    return {f for f in os.listdir(claim_dir(out_dir)) if f != '.keep'}


def claim_gene(out_dir: str, gene: str) -> bool:
    """O_EXCL 原子认领（跨机 NFS 互斥）。"""
    p = os.path.join(claim_dir(out_dir), gene)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{HOST}:{os.getpid()}:{time.time()}\n'.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_claim(out_dir: str, gene: str) -> None:
    try:
        os.remove(os.path.join(claim_dir(out_dir), gene))
    except FileNotFoundError:
        pass


def cleanup_stale_claims(out_dir: str) -> None:
    """启动清理：本机死 pid 的遗留 claim、>24h 的陈旧 claim。"""
    now = time.time()
    for f in list(claimed_genes(out_dir)):
        p = os.path.join(claim_dir(out_dir), f)
        try:
            host, pid, ts = open(p).read().strip().split(':')
        except Exception:
            os.remove(p)
            continue
        stale = (now - float(ts)) > 86400
        dead_local = (host == HOST and _pid_dead(int(pid)))
        if stale or dead_local:
            os.remove(p)
            print(f'[claim] 清理陈旧认领 {f} ({host}:{pid})', flush=True)


def _pid_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint_path', required=True)
    ap.add_argument('--corpus_paths', nargs='*', default=[],
                    help='训练语料文件列表（传给 worker 派生缓存/mask/vocab 键，须与训练时一致）')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--heldout_line', default='RPE1')
    ap.add_argument('--gpus', type=int, default=7)
    ap.add_argument('--ode_steps', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=5)
    ap.add_argument('--free_mb', type=int, default=51200, help='空闲显存门控（默认 50G）')
    ap.add_argument('--poll_s', type=float, default=10.0, help='轮询周期（秒）')
    ap.add_argument('--max_retry', type=int, default=3)
    ap.add_argument('--settle_s', type=float, default=0.0,
                    help='启动后先等待 N 秒再扫描 done/下发（供重启时等在飞旧 worker 跑完）')
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    out_dir = os.path.abspath(args.out_dir)

    # 单实例锁（按 host 分离，跨机两实例各持一把）
    lock = os.path.join(out_dir, f'.dispatch.pid.{HOST}')
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            print(f'[abort] 本机已有 dispatcher 在跑 (pid={pid}, {lock})', flush=True)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            pass
    open(lock, 'w').write(str(os.getpid()))

    if args.settle_s > 0:
        print(f'[dispatch:{HOST}] 等待在飞 worker 完成 {args.settle_s:.0f}s 后再扫描', flush=True)
        time.sleep(args.settle_s)

    cleanup_stale_claims(out_dir)

    # worker 日志目录 = logs/<out_dir 基名>/dispatch（用户 2026-09-26 定案：
    # logs/<task>_<YYYY-MM-DD_HH-MM>/ 每轮一个日期目录，与 output 同名目录对应）
    log_dir = os.path.join('logs', os.path.basename(out_dir), 'dispatch')
    os.makedirs(log_dir, exist_ok=True)

    # dispatcher 自身日志（同规范，不依赖启动方 shell 重定向）
    dlog_path = os.path.join(os.path.dirname(log_dir), f'dispatcher_{HOST}.log')
    dlog = open(dlog_path, 'a', buffering=1)
    os.dup2(dlog.fileno(), 1)
    os.dup2(dlog.fileno(), 2)
    sys.stdout = dlog
    sys.stderr = dlog
    print(f'[dispatch:{HOST}] self-log: {dlog_path}', flush=True)

    with open(os.path.join(out_dir, 'perts.txt')) as f:
        all_genes = [l.strip() for l in f if l.strip()]
    done = load_done(out_dir)
    held = claimed_genes(out_dir)
    queue = [g for g in all_genes if g not in done and g not in held]
    print(f'[dispatch:{HOST}] {len(queue)} 待推理 / {len(done)} 已完成 / '
          f'{len(held)} 他机认领中（共 {len(all_genes)}）', flush=True)

    map_path = os.path.join(out_dir, f'dispatch_map_{HOST}.txt')
    task_next = 0
    if os.path.exists(map_path):
        with open(map_path) as f:
            task_next = sum(1 for _ in f)
    map_f = open(map_path, 'a')

    procs: dict[int, tuple[int, str, subprocess.Popen]] = {}  # gpu -> (task, gene, proc)
    retry: dict[str, int] = {}
    gave_up: list[str] = []

    def spawn(g: int, gene: str) -> None:
        nonlocal task_next
        task = task_next
        task_next += 1
        map_f.write(f'{task}\t{gene}\n')
        map_f.flush()
        log = os.path.join(log_dir, f'{HOST}_g{g:02d}_t{task:03d}_{gene}.log')
        cmd = [
            os.path.join(root, '.venv/bin/python'), '-m',
            'src.script.benchmark_line_holdout',
            '--checkpoint_path', args.checkpoint_path,
            *(['--corpus_paths', *args.corpus_paths] if args.corpus_paths else []),
            '--heldout_line', args.heldout_line,
            '--out_dir', out_dir,
            '--no_eval', '--reuse_real',
            '--batch_size', str(args.batch_size),
            '--ode_steps', str(args.ode_steps),
            '--seed', str(42 + task),
            '--perts', gene,
            '--pred_tag', f'shard_disp_{HOST}_{task}',
            '--parts_only',
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(g)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        proc = subprocess.Popen(cmd, env=env, stdout=open(log, 'w'),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)  # 脱离 dispatcher 进程组：
        # dispatcher/tmux 被杀不连坐 worker，在飞基因跑完照常落盘
        procs[g] = (task, gene, proc)
        print(f'[dispatch:{HOST}] gpu{g} <- t{task} {gene} '
              f'({time.strftime("%H:%M:%S")})', flush=True)

    while queue or procs:
        free = free_mem_mib()
        # 收割已退出 worker
        for g, (task, gene, proc) in list(procs.items()):
            if proc.poll() is not None:
                part = os.path.join(out_dir, 'predparts',
                                    f'pred_shard_disp_{HOST}_{task}_g000.h5ad')
                ok = os.path.exists(part)
                del procs[g]
                if ok:
                    release_claim(out_dir, gene)
                    done.add(gene)
                    print(f'[dispatch:{HOST}] gpu{g} t{task} {gene} 完成 '
                          f'({len(done)}/{len(done) + len(queue)})', flush=True)
                else:
                    release_claim(out_dir, gene)
                    retry[gene] = retry.get(gene, 0) + 1
                    if retry[gene] > args.max_retry:
                        gave_up.append(gene)
                        print(f'[ERROR] {gene} 失败 {retry[gene]} 次，放弃 '
                              f'（日志 logs/dispatch/{HOST}_g{g:02d}_t{task:03d}_{gene}.log）',
                              flush=True)
                    else:
                        queue.append(gene)
                        print(f'[retry:{HOST}] {gene} 失败（第 {retry[gene]} 次），重新入队',
                              flush=True)
        # 空闲（可用 > 门控）且无我方进程的卡领任务（跨机 O_EXCL 认领）
        for g in range(args.gpus):
            if g in procs:
                continue
            if free.get(g, 0) > args.free_mb:
                for i in range(len(queue)):
                    gene = queue[i]
                    if claim_gene(out_dir, gene):
                        del queue[i]
                        spawn(g, gene)
                        break
        time.sleep(args.poll_s + random.uniform(0, 2.0))  # 跨机错峰：NFS O_EXCL
        # 在极窄竞争窗下偶发双认领（2026-09-24 实测 ACIN1），加随机抖动把两机
        # 的认领时刻错开；偶发重复基因无害（eval 多 400 细胞同模型样本）

    map_f.close()
    os.remove(lock)
    if gave_up:
        print(f'[dispatch:{HOST}] 失败放弃 {len(gave_up)}: {gave_up}', flush=True)

    # 最终 eval：跨机 O_EXCL 认领 + 完整性等待（他机仍在飞时轮询到齐再收口）
    eval_claim = os.path.join(out_dir, '.eval.claim')
    try:
        fd = os.open(eval_claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{HOST}:{os.getpid()}:{time.time()}\n'.encode())
        os.close(fd)
    except FileExistsError:
        print(f'[dispatch:{HOST}] 最终 eval 由他机负责，本机退出', flush=True)
        sys.exit(0)
    while len(load_done(out_dir)) < len(all_genes) - len(gave_up):
        print(f'[dispatch:{HOST}] 等待他机收尾：{len(load_done(out_dir))}/'
              f'{len(all_genes) - len(gave_up)}', flush=True)
        time.sleep(120)
    print(f'[dispatch:{HOST}] 全部基因完成，启动最终 eval_only', flush=True)
    subprocess.run([
        os.path.join(root, '.venv/bin/python'), '-m',
        'src.script.benchmark_line_holdout',
        '--checkpoint_path', args.checkpoint_path,
        '--heldout_line', args.heldout_line,
        '--out_dir', out_dir,
        '--eval_only',
    ], check=True)
    print(f'[dispatch:{HOST}] 完成：最终 scores.csv = {os.path.join(out_dir, "scores.csv")}',
          flush=True)


if __name__ == '__main__':
    main()
