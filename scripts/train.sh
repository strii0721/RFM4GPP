#!/usr/bin/env bash
# VCC-2026 scDFM 训练。远程服务器【项目根目录】执行，先激活 uv 虚拟环境：
#   source .venv/bin/activate
#
# 用法:
#   bash scripts/train.sh              # 默认 8 卡 DDP（torchrun）
#   GPUS=1 bash scripts/train.sh       # 单卡
#   STEPS=200 bash scripts/train.sh    # 覆盖步数
#   TOP_INFER=11919 bash scripts/train.sh  # 训练每步建模基因数 L（默认 11919=全轴；min 到缓存列 11,371）
# 训练超参走 config/config.py 默认值（VCC 主线已收口）；训练语料必须显式传入
# （2026-09-27 定案：不再硬编码数据集，缓存/mask/vocab 派生键含语料文件名）:
#   bash scripts/train.sh --corpus_paths /path/a.h5ad /path/b.h5ad
# 其他覆盖直接追加 tyro 参数:
#   bash scripts/train.sh --gamma=1.0 --max_test_perts=0
# 产物: output/train_{YYYY-MM-DD_HH-MM}/iteration_N/checkpoint.pt（见 config.make_path，时间戳即实验名）
set -euo pipefail
export PYTHONPATH=.
# venv 自激活：ssh 非交互 shell / tmux 不 source .bashrc，裸 python/torchrun 不在
# PATH（2026-09-21 实测 exit 127）。存在 .venv 时把它的 bin 前置（幂等，已激活也无害）。
if [ -d ".venv/bin" ]; then
  export PATH="$(pwd)/.venv/bin:$PATH"
fi
# 管道/非 tty 下 python stdout 默认 8KB 块缓冲 → tmux/日志看不到实时进度；
# 置非缓冲（stderr 本就逐行落盘），训练打印（config dump/checkpoint/进度）即时可见
export PYTHONUNBUFFERED=1
# 2026-09-22：全轴 L=11,371 时 B=2 反向峰值 ~73GB+7.7GB 瞬时分配；不带本变量时
# CUDA 缓存分配器碎片浪费 ~9.8GB reserved-unallocated → 越 80GB 墙 OOM（实测
# rank3 backward OOM：72.85 在用 + 需 7.71）。expandable_segments 让空闲段可
# 合并复用（torch 2.7 支持），探针实测 B=2=71GB 即含本变量。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 日志/输出统一阶段+时间戳目录（2026-09-26 用户定案）：logs/train_{ts}/ + output/train_{ts}/，
# ts=任务启动时刻；make_path 读同一变量（SCDFM_RUN_TS）保证二者同名对应。
export SCDFM_RUN_TS="${SCDFM_RUN_TS:-$(date +%F_%H-%M)}"
LOG_DIR="logs/train_${SCDFM_RUN_TS}"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/train.log") 2>&1

GPUS="${GPUS:-8}"
# 2026-09-21：全轴缓存每 rank RAM ~61GB（int64→int32 索引后）。共享机 1TB 内存
# + 他人作业占用，8 rank ≈490GB 可能越可用内存；空卡不足时用
# GPUS=7 + CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 这类方式选空卡。

# 论文全局 batch=96（附录 A.4.3）；多卡 DDP 时按卡数分摊到每 rank。
# 2026-09-20 全轴 L=11,071 + 梯度检查点（model.py 正向循环）：探针实测单卡
# 峰值 B=1→36GB、B=2→71GB、B=3→82GB(越 80GB 墙，含 expandable_segments)。
# → RANK_BATCH 上限=2，默认 8 卡 BATCH_TOTAL=16；单卡时请 BATCH_TOTAL=2。
BATCH_TOTAL="${BATCH_TOTAL:-16}"
RANK_BATCH=$((BATCH_TOTAL / GPUS))

# 训练每步建模基因数 L（2026-09-21 定案：建模基因子集 = 完整基因轴，含 300 panel；
# 推理侧仅对扰动自身靶列置 0。每步从采样池随机抽 L——池默认 = config.train_pool_path
# 的清单（∩ 语料 var）；空串回退 = 整个基因轴（缓存列 11,371，含 panel）。
# 缓存/mask/vocab 派生键含池路径，换池须重建。
TOP_INFER="${TOP_INFER:-11919}"

ARGS=(--data_name=vcc --batch_size="$RANK_BATCH" --infer_top_gene="$TOP_INFER")
[ -n "${STEPS:-}" ] && ARGS+=(--steps="$STEPS")
[ -n "${PRINT_EVERY:-}" ] && ARGS+=(--print_every="$PRINT_EVERY")

if [ "$GPUS" -gt 1 ]; then
  # 预处理缓存/共表达 mask/vocab 必须先单进程预构建：8 个 DDP rank 并发写
  # 同一 h5ad 会在 NFS 上撞 h5py 文件锁（实测 BlockingIOError errno 11，
  # 缓存写坏 + 全体退出）。预构建后各 rank 只读。
  python src/script/build_vcc_cache.py "${ARGS[@]}" "$@"
  # DDP：循环内 eval 会 NCCL 死锁，config 默认 do_eval=False，这里再显式关一次
  torchrun --nproc_per_node="$GPUS" src/script/run.py "${ARGS[@]}" --no-do_eval "$@"
else
  python src/script/run.py "${ARGS[@]}" "$@"
fi
