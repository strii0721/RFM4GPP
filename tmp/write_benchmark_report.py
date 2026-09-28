#!/usr/bin/env python3
"""把 cell-eval2 官方三件套产物汇总成 6 指标 raw+scaled 表格 md（用户定案格式）。

输入（benchmark_line_holdout.py 产出）:
  <out_dir>/scores.csv          -- {metric, from_baseline, ...}，末行 avg_score
  <out_dir>/run/agg_results.csv -- {statistic, <指标列...>}，raw 取 statistic=mean 行
输出:
  <out_dir>/report_<YYYY-MM-DD_HH-MM>.md（报告随任务目录，2026-09-26 目录规范）
  两行表头（Overall | PDSsc | MSEsc | JACsc | NMAEsc | FIDsc | REACHsc，每指标
  Scaled/Raw 两列）+ 每模型一行数据，不写注释行。
  Scaled = scores.csv 的 from_baseline；Overall = avg_score 行 from_baseline；
  Raw = agg_results.csv mean 行对应指标列。

用法（远程项目根）:
  .venv/bin/python -u src/script/write_benchmark_report.py \
      --out_dir output/benchmark_2026-09-26_14-00 --line RPE1 [--model_name <名>]
"""
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import tyro

# 表格列序与 vcc2026 preset 六指标（canonical 名，cell-eval2 0.16.0 catalog）
DISPLAY_COLS = [
    ('PDSsc', 'pds_cosine'),
    ('MSEsc', 'expr_mse_unbiased_capped_norm'),
    ('JACsc', 'de_wilcoxon_sig_jaccard'),
    ('NMAEsc', 'de_wilcoxon_lfc_nmae'),
    ('FIDsc', 'de_wilcoxon_direction_fidelity_yield_raw'),
    ('REACHsc', 'de_wilcoxon_direction_reach_raw'),
]
# v1 别名兜底（cell-eval2 接受别名输入；agg 列偶以别名拼写出现时仍能对上）
ALIASES = {
    'discrimination_score_cosine': 'pds_cosine',
}


def _canon(name: str) -> str:
    return ALIASES.get(name, name)


def _fmt(v: float, nd: int) -> str:
    """数字 → 用户格式（Unicode 负号，NaN → 空）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '—'
    return f'{v:.{nd}f}'.replace('-', '\u2212')


@dataclass
class Args:
    out_dir: str
    line: str = 'RPE1'                 # 数据行标签（= benchmark 的 context 标签）
    model_name: str = ''               # 留空则行标签 = line
    benchmark_dir: str = ''  # md 落点（空=out_dir 自身；2026-09-26 目录规范：报告随任务目录）


def main() -> None:
    args = tyro.cli(Args)
    scores_path = os.path.join(args.out_dir, 'scores.csv')
    agg_path = os.path.join(args.out_dir, 'run', 'agg_results.csv')
    for p, what in [(scores_path, 'scores.csv'), (agg_path, 'run/agg_results.csv')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'{what} missing: {p}（先跑 benchmark 三件套）')

    scores = pd.read_csv(scores_path)
    agg = pd.read_csv(agg_path)

    # scaled：metric 行 → from_baseline；canonical 化后建索引
    canon_scores = {}
    for _, row in scores.iterrows():
        m = _canon(str(row['metric']))
        v = row.get('from_baseline', None)
        canon_scores[m] = float(v) if pd.notna(v) else None
    if 'avg_score' not in canon_scores:
        raise ValueError(f'scores.csv 无 avg_score 行（末行），got metrics={list(canon_scores)}')
    overall = canon_scores['avg_score']

    # raw：agg mean 行；canonical 化列名
    if 'statistic' not in agg.columns:
        raise ValueError(f'agg_results.csv 无 statistic 列: {list(agg.columns)}')
    mean_rows = agg[agg['statistic'] == 'mean']
    if len(mean_rows) != 1:
        raise ValueError(f'agg_results.csv 应有恰好 1 行 statistic=mean，got {len(mean_rows)}')
    raw_row = mean_rows.iloc[0]
    canon_raw = {_canon(str(c)): float(raw_row[c]) for c in agg.columns if c != 'statistic'}

    cells = []
    for disp, canon in DISPLAY_COLS:
        if canon not in canon_scores:
            raise ValueError(
                f'scores.csv 缺指标 {canon}（可用: {sorted(canon_scores)}）'
                f'——若 cell-eval2 换了列名拼写，在 ALIASES 补一条映射')
        if canon not in canon_raw:
            raise ValueError(
                f'agg_results.csv 缺指标列 {canon}（可用: {sorted(canon_raw)}）'
                f'——若 cell-eval2 换了列名拼写，在 ALIASES 补一条映射')
        cells.append(_fmt(canon_scores[canon], 4))
        cells.append(_fmt(canon_raw[canon], 3))

    label = args.model_name or args.line
    header1_cells = ['', 'Overall'] + sum([[d, ''] for d, _ in DISPLAY_COLS], [])
    header2_cells = ['', ''] + sum([['Scaled', 'Raw'] for _ in DISPLAY_COLS], [])
    row_cells = [label, _fmt(overall, 4)] + cells
    n_cols = len(header1_cells)
    assert n_cols == len(header2_cells) == len(row_cells) == 14, f'unexpected {n_cols} cols'

    tag = os.path.basename(os.path.normpath(args.out_dir))
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
    md_dir = args.benchmark_dir or args.out_dir  # 2026-09-26 定案：报告落任务目录内
    os.makedirs(md_dir, exist_ok=True)
    md_path = os.path.join(md_dir, f'report_{ts}.md')
    with open(md_path, 'w') as f:
        f.write(f'# {label} Benchmark（{datetime.now().strftime("%Y-%m-%d")}）\n\n')
        f.write('| ' + ' | '.join(header1_cells) + ' |\n')
        f.write('|' + '---|' * n_cols + '\n')
        f.write('| ' + ' | '.join(header2_cells) + ' |\n')
        f.write('| ' + ' | '.join(row_cells) + ' |\n')
    print(f'report written: {md_path}', flush=True)
    print(f'overall(scaled)={_fmt(overall, 4)} | ' +
          ' '.join(f'{d}={cells[2 * i]}/{cells[2 * i + 1]}'
                   for i, (d, _) in enumerate(DISPLAY_COLS)), flush=True)


if __name__ == '__main__':
    main()
