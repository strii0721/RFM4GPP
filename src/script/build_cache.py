"""单独构建训练缓存三件套（processed 缓存 + 共表达 mask + vocab；2026-09-28 定案）。

从 train.py rank0 预建逻辑拆出（原 build_vcc_cache.py 职责）：
  1. 全量读入 common.train_set_paths（derive_pert_columns 语义列匹配
     + perturb_direction 扰动方向过滤）
  2. CP10k + log1p + HVG 列选择 → processed_*.h5ad（缓存）+ .meta
     （meta 供 build_tensors.py 做张量列对齐）
  3. 共表达图 mask_*.pt + vocab json

顺序约束：先跑本脚本（缓存 meta 落盘）→ 再跑 build_tensors.py（张量列对齐
依赖缓存列）→ 最后 train.py（rank0 检测缓存已存在跳过预建）。

用法（远程项目根）：
    .venv/bin/python src/script/build_cache.py
日志：<log_base_dir>/cache_<ts>/build.log
"""
import datetime
import os
import sys
import time

from src.data_process.data import Data
from src.utils.config_utils import ConfigUtils, FlowConfig
from src.utils.utils import process_vocab


def build_cache(cfg):
    """单进程构建缓存 + mask + vocab；返回构建好的 Data（train.py rank0 预建复用）。"""
    t0 = time.time()
    d = Data(cfg.data_path, config=cfg)
    d.load_data(cfg.data_name)
    d.process_data(
        n_top_genes=cfg.n_top_genes, infer_top_gene=cfg.infer_top_gene,
        split_method=cfg.split_method, fold=cfg.fold,
        use_negative_edge=cfg.use_negative_edge, k=cfg.topk,
    )
    process_vocab(d, cfg)
    print(f'cache+mask+vocab ready in {time.time()-t0:.0f}s '
          f'(cache={cfg.train_cache_dir} mask={d.mask_path})', flush=True)
    return d


def main() -> None:
    cfg = ConfigUtils.load(FlowConfig, use_cli=False)
    # 2026-10-03 调试：25M 行 obs 操作某些步骤实测卡死（无 traceback、CPU 满），
    # 定时 dump 当前 Python 栈到日志（stderr 已重定向）定位卡点；正常结束前取消。
    import faulthandler
    faulthandler.dump_traceback_later(300, exit=False)
    ts = os.environ.get('SCDFM_RUN_TS') or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_dir = os.path.join(cfg.log_base_dir, f'cache_{ts}')
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(os.path.join(log_dir, 'build.log'), 'a', buffering=1)
    os.dup2(log_f.fileno(), 1)
    os.dup2(log_f.fileno(), 2)
    sys.stdout = log_f
    sys.stderr = log_f
    build_cache(cfg)


if __name__ == '__main__':
    main()
