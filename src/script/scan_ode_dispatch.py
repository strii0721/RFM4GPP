#!/usr/bin/env python3
"""ODE 收敛性扫描分发器（2026-09-26）：把 (gene, N) 任务按 GPU 空闲度分发。

任务 = scan_ode_convergence.py 任务模式（一进程一 (gene, N)），产物
{out_dir}/{gene}_n{N}.npy；已存在跳过（resume）。全跑完跑报告模式出推荐 N。

用法（远程项目根）:
  .venv/bin/python -u src/script/scan_ode_dispatch.py \
      --checkpoint_path output/train_<ts>/iteration_N/checkpoint.pt \
      --real_path output/benchmark_2026-09-24_19-09/real.h5ad \
      --genes ACIN1,ARID5B,BRD4 --n_list 12,25,50,100,200 \
      --out_dir output/scan_<ts> --gpus 8

日志：dispatcher 自建 logs/<out_dir 基名>/dispatcher_<host>.log，
worker 落 logs/<out_dir 基名>/dispatch/<host>_g<gpu>_<gene>_n<N>.log（目录规范）。
"""
import argparse
import os
import socket
import subprocess
import sys
import time

HOST = socket.gethostname().split('.')[0]


def free_mem_mib() -> dict[int, int]:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    return {int(x.strip()): int(t) - int(u)
            for x, t, u in (line.split(',') for line in out.strip().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint_path', required=True)
    ap.add_argument('--real_path', required=True)
    ap.add_argument('--genes', required=True, help='逗号分隔 panel 基因（扫描子集）')
    ap.add_argument('--n_list', default='12,25,50,100,200')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gpus', type=int, default=8)
    ap.add_argument('--free_mb', type=int, default=51200, help='空闲显存门控（默认 50G）')
    ap.add_argument('--batch_size', type=int, default=5)
    ap.add_argument('--max_retry', type=int, default=2)
    ap.add_argument('--poll_s', type=float, default=20.0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('logs', os.path.basename(out_dir), 'dispatch')
    os.makedirs(log_dir, exist_ok=True)
    dlog = open(os.path.join(os.path.dirname(log_dir), f'dispatcher_{HOST}.log'),
                'a', buffering=1)
    os.dup2(dlog.fileno(), 1)
    os.dup2(dlog.fileno(), 2)
    sys.stdout = dlog
    sys.stderr = dlog
    print(f'[scan:{HOST}] self-log: {dlog.name}', flush=True)

    genes = [g.strip() for g in args.genes.split(',') if g.strip()]
    n_list = sorted((int(n) for n in args.n_list.split(',')), reverse=True)  # 长任务先发
    queue = [(g, n) for n in n_list for g in genes]
    done = {t for t in queue if os.path.exists(os.path.join(out_dir, f'{t[0]}_n{t[1]}.npy'))}
    pending = [t for t in queue if t not in done]
    print(f'[scan:{HOST}] {len(queue)} tasks: {len(done)} done, {len(pending)} pending', flush=True)

    procs: dict[int, tuple[tuple, subprocess.Popen]] = {}
    retry: dict[tuple, int] = {}
    while pending or procs:
        for g in range(args.gpus):
            if g in procs:
                continue
            if not pending:
                break
            if free_mem_mib().get(g, 0) < args.free_mb:
                continue
            task = pending.pop(0)
            gene, n = task
            log = os.path.join(log_dir, f'{HOST}_g{g:02d}_{gene}_n{n}.log')
            with open(log, 'a') as lf:
                cmd = [sys.executable, '-u', 'src/script/scan_ode_convergence.py',
                       '--checkpoint_path', args.checkpoint_path,
                       '--real_path', args.real_path,
                       '--gene', gene, '--n_steps', str(n),
                       '--out_dir', out_dir, '--gpu', str(g),
                       '--batch_size', str(args.batch_size)]
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            procs[g] = (task, proc)
            print(f'[scan:{HOST}] spawn g{g}: {gene} n={n} ({len(pending)} left)', flush=True)
        time.sleep(args.poll_s)
        for g in list(procs):
            task, proc = procs[g]
            if proc.poll() is None:
                continue
            gene, n = task
            part = os.path.join(out_dir, f'{gene}_n{n}.npy')
            if proc.returncode == 0 and os.path.exists(part):
                print(f'[scan:{HOST}] done g{g}: {gene} n={n}', flush=True)
            else:
                retry[task] = retry.get(task, 0) + 1
                if retry[task] <= args.max_retry:
                    pending.append(task)
                    print(f'[scan:{HOST}] retry {retry[task]}/{args.max_retry}: '
                          f'{gene} n={n} (rc={proc.returncode})', flush=True)
                else:
                    print(f'[scan:{HOST}] GAVE UP {gene} n={n}', flush=True)
            del procs[g]

    print(f'[scan:{HOST}] all tasks finished. 报告模式：', flush=True)
    print(f'  {sys.executable} -u src/script/scan_ode_convergence.py --report 1 '
          f'--genes {args.genes} --n_list {args.n_list} --out_dir {out_dir}', flush=True)


if __name__ == '__main__':
    main()
