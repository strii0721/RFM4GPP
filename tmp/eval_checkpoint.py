#!/usr/bin/env python3
"""Evaluate a saved scDFM checkpoint (single GPU), line-consistent.

Reuses run.py's test() with its globals set. The test set is the held-out
genes of split_method='single' (80/20 gene holdout); each (gene, cell line)
pair is scored against its own line's control/target pools, then averaged
over lines (see TestDataset.perturbation_line_pairs).

Usage:
  python src/script/eval_checkpoint.py --checkpoint_path <checkpoint.pt> \
      [same vcc data args as training] [--max_test_perts 20]
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 2026-09-28: tmp/ 下减一层

import torch
import tyro
from accelerate import Accelerator

import src.script.train as runner
from src.utils.config_utils import ConfigUtils, FlowConfig
from src.data_process.data import Data
from src.models.instantiate_model import instantiate_model
from src.utils.utils import process_vocab


@dataclass
class _EvalCfg(FlowConfig):
    checkpoint_path: str = ''   # 2026-10-06：FlowConfig 字段删除后本地自带（CLI 传）


def main():
    config = ConfigUtils.load(_EvalCfg, description=__doc__)
    assert config.checkpoint_path and os.path.exists(config.checkpoint_path), "checkpoint_path required"

    accelerator = Accelerator()
    device = accelerator.device
    runner.config = config
    runner.device = device

    # data (same pipeline as training)
    data_manager = Data(config.data_path, config=config)
    data_manager.load_data(config.data_name)
    data_manager.process_data(
        n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene,
        split_method=config.split_method, fold=config.fold,
        use_negative_edge=config.use_negative_edge, k=config.topk,
    )
    _, valid_sampler, _ = data_manager.load_flow_data(batch_size=config.batch_size)

    # model + checkpoint (weights only)
    mask_path = data_manager.mask_path
    vf = instantiate_model(
        config.model_type, ntoken=config.ntoken, d_model=config.d_model,
        d_perturbation=config.d_model, fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function, mask_path=mask_path,
    )
    ckpt = torch.load(config.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device)

    vocab = process_vocab(data_manager, config)
    # globals referenced by run.py's test()
    runner.inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}

    out_dir = os.path.join(os.path.dirname(config.checkpoint_path), 'eval')
    os.makedirs(out_dir, exist_ok=True)
    score = runner.test(valid_sampler, vf, accelerator, batch_size=config.batch_size,
                        path=out_dir, vocab=vocab, scheme='composite')
    print(f"eval done, score={score} (line-consistent mean over lines)")


if __name__ == "__main__":
    main()
