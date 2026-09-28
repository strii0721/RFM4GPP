#!/usr/bin/env bash
# 本地六指标评分（2026-09-28 定案）：读取已生成的预测 h5ad，与真实对照 h5ad
# 对齐后跑官方三件套（baseline/run/score，cell-eval2 vcc2026 preset）。
# 远程服务器【项目根目录】执行（先 source .venv/bin/activate 或设 PY）。
#
# 用法（flag 指定所需变量）:
#   bash scripts/local_benchmark.sh --pred_h5ad <p.h5ad> --real_h5ad <r.h5ad> --out_dir <dir>
# 环境变量:
#   PY         python 解释器（默认项目 .venv/bin/python）
#   EVAL_CMD   cell-eval2 命令（默认 eval 2>/dev/null 查 PATH）
set -euo pipefail

PY="${PY:-.venv/bin/python}"
PRED_H5AD=""
REAL_H5AD=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred_h5ad) PRED_H5AD="$2"; shift 2 ;;
    --real_h5ad) REAL_H5AD="$2"; shift 2 ;;
    --out_dir)   OUT_DIR="$2";   shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

[ -n "$PRED_H5AD" ] || { echo "missing: --pred_h5ad" >&2; exit 1; }
[ -n "$REAL_H5AD" ] || { echo "missing: --real_h5ad" >&2; exit 1; }
[ -n "$OUT_DIR" ]   || { echo "missing: --out_dir" >&2; exit 1; }
[ -f "$PRED_H5AD" ] || { echo "missing file: $PRED_H5AD" >&2; exit 1; }
[ -f "$REAL_H5AD" ] || { echo "missing file: $REAL_H5AD" >&2; exit 1; }

export OUT_DIR REAL_PATH="$REAL_H5AD" PRED_PATH="$PRED_H5AD"
"$PY" -u tmp/eval_external_pred.py
echo "done: $OUT_DIR/scores.csv"
