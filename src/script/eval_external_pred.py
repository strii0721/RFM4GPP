#!/usr/bin/env python
"""对外部预测文件跑 vcc2026 六指标评测（不经过模型推理，直接喂现成 pred h5ad）。

用法：
    python -m src.script.eval_external_pred \
        --real_path output/benchmark_2026-09-24_19-09/real.h5ad \
        --pred_path /ssd3/ict1/deltatransdenoise/outputs/rpe1_prediction.h5ad \
        --out_dir output/benchmark_external_<ts> --n_real_cells 100

与 benchmark_line_holdout 的 eval_only 同口径：
- real 侧：已有 real.h5ad（300 panel 真实扰动细胞 + 对照）
- pred 侧：外部文件，收口到与 real 相同的扰动集；补对照类别
- eval 前 pred 每扰动抽到 n_real_cells（与 real 侧对齐）
- 三件套：cell-eval2 baseline -> run --anchor -> score
产物：out_dir/scores.csv（raw+scaled）、real.h5ad、pred.h5ad、baseline/、run/
"""
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.script.benchmark_line_holdout import BenchConfig, _run_eval, _subsample_pred

if ad.settings and hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True


def main() -> None:
    cfg = BenchConfig(
        checkpoint_path='unused',
        out_dir=os.environ['OUT_DIR'],
    )
    real_path = os.environ['REAL_PATH']
    pred_src = os.environ['PRED_PATH']
    os.makedirs(cfg.out_dir, exist_ok=True)

    real = sc.read_h5ad(real_path)
    pred = sc.read_h5ad(pred_src)
    print(f'loaded real {real.shape} pred {pred.shape}', flush=True)

    # pred 侧按 real 的扰动集收口（两侧扰动集必须逐项一致）
    tg_real = set(real.obs['target_gene'].astype(str).unique()) - {'non-targeting'}
    tg_pred = pred.obs['target_gene'].astype(str).to_numpy()
    missing = tg_real - set(tg_pred)
    assert not missing, f'pred 缺少 {len(missing)} 个 real 扰动: {sorted(missing)[:10]}...'
    keep = np.isin(tg_pred, list(tg_real))
    pred = pred[keep].copy()
    print(f'pred narrowed to {pred.shape[0]} cells ({len(tg_real)} genes)', flush=True)

    # 对齐基因轴：pred 列重排到 real 的 var 序
    if not np.array_equal(pred.var_names.astype(str).to_numpy(),
                          real.var_names.astype(str).to_numpy()):
        pred = pred[:, real.var_names.astype(str)].copy()
        print('var reordered to real axis', flush=True)

    # 补对照类别（pred 无 non-targeting 行；官方口径两侧扰动集一致）
    ctl_mask = real.obs['target_gene'].astype(str).to_numpy() == 'non-targeting'
    ctl = real[ctl_mask].copy()
    pred = ad.concat([pred, ctl], join='outer', index_unique=None)
    print(f'after +ctl: {pred.shape[0]} cells', flush=True)

    pred = _subsample_pred(pred, cfg.n_real_cells, cfg.seed)
    # _run_eval 的 baseline CLI 需要 out_dir/real.h5ad（eval_only 流程里 real 本来就在 out_dir）
    real.write_h5ad(os.path.join(cfg.out_dir, 'real.h5ad'))
    _run_eval(cfg, real, pred)


if __name__ == '__main__':
    main()
