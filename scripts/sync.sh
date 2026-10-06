#!/usr/bin/env bash
# 同步本地 scDFM 代码到远程 vcc-worker02 部署副本（远程无 .git，代码镜像）。
#
# 用法:
#   scripts/sync_to_remote.sh             # 真实同步
#   DRY_RUN=1 scripts/sync_to_remote.sh   # 只列出将传输的文件，不落盘
#
# 排除规则: 显式排除（仓库约定大目录）+ .gitignore 逐目录规则。
#   - .gitignore: 构建产物/缓存/__pycache__/venv/assets 等（rsync 逐目录合并）
#   - data/result/output/cache/.git/.venv: 仓库约定 —— 语料在远程
#     /home/ict2/Projects/vcc-2026/resources/datasets；cache=训练缓存
#     （processed/mask/split 产物，相对项目根）；训练输出在远程本地盘生成，
#     一律不随代码同步。
# 远程目标: ict2@211.137.21.49:/home/ict2/Projects/scDFM/（2026-09-20 IP 更换 + 项目迁家目录；2026-09-22 再换 IP）
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ict2@211.137.21.49}"
REMOTE_DIR="${REMOTE_DIR:-/home/ict2/Projects/scDFM}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RSYNC_OPTS=(
  -az
  --partial
  --delete
  # 显式排除（置于 .gitignore 规则之前，先匹配先生效）
  --exclude=.git/
  --exclude=data/
  --exclude=result/
  --exclude=output/
  # cache 勿带尾斜杠：远程 cache 是软链（→/ssd4/ict2/cache），rsync 尾斜杠模式
  # 只匹配目录不匹配软链 → --delete 会删掉远程软链（2026-10-06 事故根因）。
  --exclude=cache
  --exclude=logs/
  --exclude=.venv/
  # 辅助脚本目录：只同步 *.py（2026-09-28 起辅助脚本统一放 tmp/，被 local/remote
  # benchmark 与 inference 引用，必须随代码同步）；tmp 数据文件不入库不传输
  --include=tmp/
  --include=tmp/*.py
  --exclude=tmp/*
  # 逐目录应用 .gitignore 规则
  --filter=':- .gitignore'
  --rsync-path="mkdir -p ${REMOTE_DIR} && rsync"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_OPTS+=( -n -v )
fi

rsync "${RSYNC_OPTS[@]}" "$REPO_ROOT/" "$REMOTE_HOST:${REMOTE_DIR}/"
echo "done: $REPO_ROOT -> $REMOTE_HOST:${REMOTE_DIR}"
