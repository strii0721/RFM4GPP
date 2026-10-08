#!/usr/bin/env bash
# 官方渠道提交壳脚本（2026-10-06 由 remote_benchmark.sh 更名，逻辑迁入 src/script/submit.py）。
# 不再接受 flag：全部配置经 yaml submit 节（pred_h5ad/out_dir 等提交前在 yaml 填好）。
#
# 用法:
#   bash scripts/submit.sh
#
# 坑: 远程实验室网络上传 GCS 会卡死（2.7GB .vcc 实测中途断）；卡住时把 .vcc
#     rsync 回家用 resume_upload 续传（见 skill vcc-2026 → references/vcc-cli-pipeline.md）。
set -euo pipefail

cd "$(dirname "$0")/.."
exec .venv/bin/python -m src.script.submit
