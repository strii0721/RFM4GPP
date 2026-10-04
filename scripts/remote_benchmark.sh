#!/usr/bin/env bash
# 官方渠道提交（2026-09-28 定案）：读取已生成的预测 h5ad → vcc prep 打包
# → submit 上传 virtualcellchallenge.org（每 UTC 日限 2 次）。合并自旧
# gen_vcc.sh + submit.sh，全部变量经 flag 传入。
# 2026-09-30 用户定案：模型输出为 19,843 轴预测，prep 前先按
# mrn19843/gene_names_vcc.csv 裁剪到官方提交轴 18,533（缺失补零、训练轴独有丢弃）。
# 远程服务器【项目根目录】执行。
#
# 用法:
#   bash scripts/remote_benchmark.sh --pred_h5ad <p.h5ad> --out_dir <基地址> \
#       [--model_name scDFM-K6] [--resume]
# flag:
#   --pred_h5ad    输入预测 h5ad（必填）
#   --out_dir      输出基地址（必填，如 output）；实际目录 = <基地址>/benchmark_<YYYY-MM-DD_HH-MM>
#   --out_vcc      输出 .vcc 路径（默认 <实际目录>/<pred 文件名 stem>.vcc）
#   --controls_dir 官方对照目录（默认 vcc-2026 resources/datasets/controls）
#   --genes_vcc    提交轴基因清单（默认 mrn19843/gene_names_vcc.csv，18,533 轴）
#   --model_name   提交模型名（默认 scDFM）
#   --resume       上传中断续传（aborted upload 锁团队槽位时必须用）
#   --no_wait      提交后不阻塞等评分（默认 --wait）
#   --dry-run      vcc prep 只校验不产出
# 日志：logs/benchmark_<ts>/prep_<predstem>.log（tee 双写控制台，2026-09-29 定案
#       与 local_benchmark.sh 同口径）。
# 坑: 远程实验室网络上传 GCS 会卡死（2.7GB .vcc 实测中途断）；卡住时把 .vcc
#     rsync 回家用 --resume 续传（见 skill vcc-2026 → references/vcc-cli-pipeline.md）。
set -euo pipefail

# vcc CLI: 非交互 shell（tmux/nohup）PATH 可能不含 conda env，显式回退
# （.36 上位于 miniconda3/envs/vcc2026_submit/bin/vcc，2026-09-29 实测确认）
VCC_BIN="${VCC_BIN:-$(command -v vcc 2>/dev/null || echo "$HOME/miniconda3/envs/vcc2026_submit/bin/vcc")}"

PRED_H5AD=""
OUT_BASE=""
OUT_VCC=""
CONTROLS_DIR="/home/ict2/Projects/vcc-2026/resources/datasets/controls"
GENES_VCC="/home/ict2/Projects/vcc-2026/resources/datasets/mrn19843/gene_names_vcc.csv"
MODEL_NAME="scDFM"
DRY=""
RESUME=0
NO_WAIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred_h5ad)    PRED_H5AD="$2"; shift 2 ;;
    --out_dir)      OUT_BASE="$2";   shift 2 ;;
    --out_vcc)      OUT_VCC="$2";    shift 2 ;;
    --controls_dir) CONTROLS_DIR="$2"; shift 2 ;;
    --genes_vcc)    GENES_VCC="$2";    shift 2 ;;
    --model_name)   MODEL_NAME="$2"; shift 2 ;;
    --resume)       RESUME=1; shift ;;
    --no_wait)      NO_WAIT=1; shift ;;
    --dry-run)      DRY="--dry-run"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

[ -n "$PRED_H5AD" ] || { echo "missing: --pred_h5ad" >&2; exit 1; }
[ -n "$OUT_BASE" ]  || { echo "missing: --out_dir" >&2; exit 1; }
[ -f "$PRED_H5AD" ] || { echo "missing file: $PRED_H5AD" >&2; exit 1; }
[ -f "$CONTROLS_DIR/gene_names.csv" ] || { echo "missing: $CONTROLS_DIR/gene_names.csv" >&2; exit 1; }
[ -f "$GENES_VCC" ] || { echo "missing: $GENES_VCC" >&2; exit 1; }

OUT_DIR="${OUT_BASE%/}/benchmark_$(date +%Y-%m-%d_%H-%M)"
mkdir -p "$OUT_DIR"
OUT_VCC="${OUT_VCC:-$OUT_DIR/$(basename "$PRED_H5AD" .h5ad).vcc}"

# 日志自落盘（2026-09-29 定案：与 local_benchmark.sh 同口径；tee 保留控制台实时输出）
LOG_DIR="logs/$(basename "$OUT_DIR")"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prep_$(basename "$PRED_H5AD" .h5ad).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# -g 要求无表头的基因列表（官方 gene_names.csv 首行是 'gene_name'）
GENE_CSV=$(mktemp /tmp/vcc_gene_names.XXXXXX.csv)
trap 'rm -f "$GENE_CSV"' EXIT
tail -n +2 "$CONTROLS_DIR/gene_names.csv" > "$GENE_CSV"

# 提交轴裁剪（2026-09-30 用户定案）：19,843 → 18,533，按 gene_names_vcc.csv 重排
CROPPED="$OUT_DIR/pred_18533.h5ad"
.venv/bin/python tmp/crop_to_vcc_axis.py --in "$PRED_H5AD" \
    --gene_csv "$GENES_VCC" --out "$CROPPED"

"$VCC_BIN" prep -i "$CROPPED" -g "$GENE_CSV" --perts "$CONTROLS_DIR/pert_counts.csv" -o "$OUT_VCC" --force $DRY
echo "done prep: $OUT_VCC"

[ -n "$DRY" ] && exit 0

ARGS=()
[ "$RESUME" = "1" ] && ARGS+=(--resume)
[ "$NO_WAIT" != "1" ] && ARGS+=(--wait)
"$VCC_BIN" submit "$OUT_VCC" -m "$MODEL_NAME" "${ARGS[@]}"
echo "done submit: $OUT_VCC (model=$MODEL_NAME)"
echo "log: $LOG_FILE"
