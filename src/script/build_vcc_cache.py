#!/usr/bin/env python3
"""Build the vcc processed cache + co-expression mask + vocab (SINGLE process).

Multi-GPU DDP training must never build these from 8 ranks concurrently: all
ranks write the same h5ad on NFS -> h5py file-lock collision -> BlockingIOError
(errno 11) and a corrupt cache (observed 2026-09-07). train.sh runs this script
once before torchrun; all ranks then only READ the artifacts.

Usage (same tyro args as training):
  python src/script/build_vcc_cache.py --data_name=vcc
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tyro

from config.config import FlowConfig
from src.data_process.data import Data
from src.utils.utils import process_vocab


def main():
    config = tyro.cli(FlowConfig, description=__doc__)
    data_manager = Data(config.data_path, config=config)
    data_manager.load_data(config.data_name)
    # builds cache (if missing) and co-expression mask (if missing)
    data_manager.process_data(
        n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene,
        split_method=config.split_method, fold=config.fold,
        use_negative_edge=config.use_negative_edge, k=config.topk,
    )
    # vocab: also written here so DDP ranks never race on the json file
    process_vocab(data_manager, config)
    stem = '+'.join(os.path.splitext(os.path.basename(str(p)))[0]
                    for p in sorted(config.corpus_paths))
    cache = os.path.join(config.data_path, config.data_name, f'processed_n{config.n_top_genes}_{stem}.h5ad')
    mask = data_manager.mask_path
    print(f'cache+mask+vocab ready: {cache} | {mask}')


if __name__ == '__main__':
    main()
