#!/bin/bash
# worker01 推理冒烟（2026-10-06）：ABC 各 2 基因、3 卡并行；ict1 用户可写输出目录
cd /share/ict2/Projects/scDFM
PY=/home/ict1/.venvs/scdfm/bin/python
DS=/share/ict2/Projects/vcc-2026/resources/datasets
B=$DS/mrnh19547
TRAIN="$B/marson_d1_rest.h5ad $B/marson_d1_stim8hr.h5ad $B/marson_d1_stim48hr.h5ad \
$B/marson_d2_rest.h5ad $B/marson_d2_stim8hr.h5ad $B/marson_d2_stim48hr.h5ad \
$B/marson_d3_rest.h5ad $B/marson_d3_stim8hr.h5ad $B/marson_d3_stim48hr.h5ad \
$B/marson_d4_rest.h5ad $B/marson_d4_stim8hr.h5ad $B/marson_d4_stim48hr.h5ad \
$B/nadig_hepg2.h5ad $B/nadig_jurkat.h5ad $B/replogle_k562_essential.h5ad \
$B/replogle_k562_gwps.h5ad $B/replogle_rpe1.h5ad $B/h1.h5ad"
CK=/share/ict2/Projects/scDFM/output/train_2026-10-06_14-51/iteration_1000
SHARE=/share/ict2/Projects/scDFM
for pair in "A 4" "B 5" "C 6"; do set -- $pair
  SCDFM_RUN_TS=smoke_abc$1 CUDA_VISIBLE_DEVICES=$2 PYTHONUNBUFFERED=1 nohup $PY -m src.script.inference --no_dispatch \
    --checkpoint_path $CK \
    --ae_encoder_ckpt $CK/encoder_1000.pt --ae_decoder_ckpt $CK/decoder_1000.pt \
    --test_set_paths $B/context_$1_n19547.h5ad \
    --panel_csv_path $DS/controls/pert_counts.csv \
    --train_set_paths $TRAIN \
    --train_cache_dir $SHARE/cache/train_cache_mrnh19547 \
    --frozen_tensors_dir $SHARE/output/frozen_tensors_mrnh19547 \
    --perts ABCD1,ACLY \
    --output_base_dir /home/ict1/inf_out --log_base_dir /home/ict1/inf_out \
    > /tmp/inf_$1.log 2>&1 &
done
echo "started 3 inference smokes"
