"""官方渠道提交（2026-10-06 由 scripts/remote_benchmark.sh 迁移为 Python 入口）：
读取已生成的预测 h5ad → 轴裁剪/幂等校验 → vcc prep 打包 → submit 上传
virtualcellchallenge.org（每 UTC 日限 2 次）。

配置全部经 yaml benchmark 节（2026-10-06 用户定案，不再接受 CLI flag）：
  benchmark.pred_h5ad     —— 输入预测 h5ad（必填，提交前在 yaml 填好）
  benchmark.output_base_dir —— 输出基地址；实际 = <基地址>/benchmark_<ts>
  benchmark.genes_vcc     —— 提交轴基因清单（空 = common.output_genes_csv，18533）

用法:
  bash scripts/submit.sh   # 或 .venv/bin/python -m src.script.submit
"""
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_utils import ConfigUtils, FlowConfig


@dataclass
class SubmitConfig(FlowConfig):
    # 提交配置（2026-10-06 定案：全部经 yaml benchmark 节读入，无 CLI）
    pred_h5ad: str = ''                 # 输入预测 h5ad（必填，提交前在 yaml 填好）
    # 输出基地址 = benchmark 节 output_base_dir（2026-10-06 拆段定案，不再独立 out_dir 字段）
    out_vcc: str = ''                   # .vcc 输出路径（默认 <实际目录>/<pred stem>.vcc）
    controls_dir: str = '/home/ict2/Projects/vcc-2026/resources/datasets/controls'
    genes_vcc: str = ''                 # 提交轴基因清单（空 = common.output_genes_csv）
    model_name: str = 'scDFM'
    resume_upload: bool = False          # 上传中断续传（aborted upload 锁团队槽位时必须用；
                                         # 字段避开 FlowConfig.resume（训练续训））
    no_wait: bool = False               # 提交后不阻塞等评分
    dry_run: bool = False               # prep 只校验不产出


def _vcc_bin() -> str:
    # 非交互 shell（tmux/nohup）PATH 可能不含 conda env（2026-09-29 实测）
    from shutil import which
    return (os.environ.get('VCC_BIN') or which('vcc')
            or f"{os.path.expanduser('~')}/miniconda3/envs/vcc2026_submit/bin/vcc")


def main() -> None:
    cfg = ConfigUtils.load(SubmitConfig, use_cli=False, extra_sections=('benchmark',))
    assert cfg.pred_h5ad and os.path.exists(cfg.pred_h5ad), 'pred_h5ad required'
    assert cfg.output_base_dir, 'output_base_dir required'
    genes_vcc = cfg.genes_vcc or cfg.output_genes_csv
    assert genes_vcc and os.path.exists(genes_vcc), f'genes_vcc missing: {genes_vcc}'
    gene_names_csv = os.path.join(cfg.controls_dir, 'gene_names.csv')
    assert os.path.exists(gene_names_csv), f'missing: {gene_names_csv}'
    pert_counts = os.path.join(cfg.controls_dir, 'pert_counts.csv')
    assert os.path.exists(pert_counts), f'missing: {pert_counts}'

    out_base = cfg.output_base_dir.rstrip('/')
    out_dir = f'{out_base}/benchmark_{time.strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(cfg.pred_h5ad).stem
    out_vcc = cfg.out_vcc or f'{out_dir}/{stem}.vcc'

    # 日志自落盘（2026-09-29 定案口径：log_base_dir/benchmark_<ts>/prep_<stem>.log）
    log_dir = os.path.join(cfg.log_base_dir, os.path.basename(out_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'prep_{stem}.log')
    print(f'log: {log_file}', flush=True)

    def _run(cmd: list[str]) -> None:
        print('$ ' + ' '.join(cmd), flush=True)
        with open(log_file, 'a') as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)

    # 提交轴裁剪/幂等校验（2026-10-06：推理链已内置 19547→18533 还原，pred 直接是
    # 18,533 提交轴，crop 按同清单重排 = 恒等；旧轴 pred 仍兼容补零/丢弃）
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    target = pd.read_csv(genes_vcc)['gene_name'].astype(str).tolist()
    a = ad.read_h5ad(cfg.pred_h5ad)
    # 只保留扰动行（2026-10-07：vcc 规则禁止 control 行；防御旧产物混入 non-targeting）
    if 'target_gene' in a.obs.columns:
        keep_rows = a.obs['target_gene'].astype(str) != 'non-targeting'
        a = a[keep_rows]
        print(f'filter non-targeting rows: {int((~keep_rows).sum())} dropped', flush=True)
    # HGNC 别名映射已废弃（2026-10-07）：TIAF1 是 MYO18A 的基因前体、非同基因，
    # 新词表 mrnh19657 中 TIAF1 独立成列，此处恢复直映射+补零
    pos = {g: i for i, g in enumerate(a.var_names.astype(str))}
    idx = np.array([pos.get(g, -1) for g in target], dtype=np.int64)
    valid = idx >= 0
    Xv = a.X[:, idx[valid]]
    if sp.issparse(Xv):
        out = sp.lil_matrix((a.n_obs, len(target)), dtype=Xv.dtype)
        out[:, valid] = Xv.tocsr()
        X = out.tocsr()
    else:
        X = np.zeros((a.n_obs, len(target)), dtype=Xv.dtype)
        X[:, valid] = Xv
    # 超限细胞总量缩放（2026-10-08 临时方案）：r̂ clip 后个别细胞总 counts > 1e6 被
    # vcc prep 拒（官方硬约束 per-cell ≤1e6）；比例缩放到 1e6 后 floor 保整数且 ≤上限
    _row_sum = np.asarray(X.sum(axis=1)).ravel()
    _over = _row_sum > 1e6
    if _over.any():
        _scale = 1e6 / _row_sum[_over]
        if sp.issparse(X):
            _Xc = X.tocsr().copy()
            _Xc[_over] = _Xc[_over].multiply(_scale[:, None])
            _Xc.data = np.floor(_Xc.data)
            X = _Xc
        else:
            _Xd = X.copy()
            _Xd[_over] = _Xd[_over] * _scale[:, None]
            X = np.floor(_Xd)
        print(f'cap per-cell counts: {int(_over.sum())} cells scaled to <=1e6 '
              f'(worst raw {_row_sum.max():.1e})', flush=True)
    cropped = f'{out_dir}/pred_18533.h5ad'
    b = ad.AnnData(X=X, obs=a.obs.copy(), var=pd.DataFrame(index=target))
    b.write_h5ad(cropped)
    print(f'crop {a.shape} -> {b.shape} '
          f'(missing zero-pad={int((~valid).sum())}, dropped={int(a.n_vars - valid.sum())})',
          flush=True)

    # vcc prep -g 要求无表头的基因列表（官方 gene_names.csv 首行是 'gene_name'）
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as gf:
        with open(gene_names_csv) as f:
            next(f)
            gf.write(f.read())
        gene_csv_tmp = gf.name
    try:
        vcc = _vcc_bin()
        dry = ['--dry-run'] if cfg.dry_run else []
        _run([vcc, 'prep', '-i', cropped, '-g', gene_csv_tmp,
              '--perts', pert_counts, '-o', out_vcc, '--force'] + dry)
        print(f'done prep: {out_vcc}')
        if cfg.dry_run:
            return
        args = []
        if cfg.resume_upload:
            args.append('--resume')
        if not cfg.no_wait:
            args.append('--wait')
        _run([vcc, 'submit', out_vcc, '-m', cfg.model_name] + args)
        print(f'done submit: {out_vcc} (model={cfg.model_name})')
        print(f'log: {log_file}')
    finally:
        os.unlink(gene_csv_tmp)


if __name__ == '__main__':
    main()
