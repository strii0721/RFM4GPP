import os
import sys

from src.utils.config_utils import ConfigUtils, FlowConfig
from tmp.generate_submission import GenConfig, artifact_paths

# 1) 无 CLI 参数：全部值来自 YAML（dataclass 无默认值）
sys.argv = ['test']
f = ConfigUtils.load(FlowConfig)
assert f.batch_size_per_gpu == 32, f.batch_size_per_gpu
assert f.steps == 300000, f.steps
assert f.early_stop_patience == 10, f.early_stop_patience
assert f.resume == '', f.resume
assert f.ntoken == 19661, f.ntoken
assert len(f.train_set_paths) == 18, len(f.train_set_paths)
assert f.train_set_paths[0].endswith('mrnh19657/marson_d1_rest.h5ad')
assert f.train_set_paths[-1].endswith('mrnh19657/h1.h5ad'), f.train_set_paths[-1]
assert f.perturb_direction == ['CRISPRi'], f.perturb_direction
assert f.output_base_dir == 'output', f.output_base_dir
assert f.log_base_dir == 'logs', f.log_base_dir
assert f.make_path('train').startswith('output/train_'), f.make_path('train')
assert f.make_path('inference').startswith('output/inference_'), f.make_path('inference')
assert (f.n_ctrl_cells, f.n_pred_cells, f.n_real_cells, f.min_real_cells) == (0, 0, 0, 0)  # 2026-10-06 拆段：训练入口默认空
assert f.test_set_paths == '', f.test_set_paths  # 2026-10-06 拆段：test_set_paths 在 inference 节，训练入口默认空
assert f.frozen_tensors_dir == '/home/ict2/Projects/scDFM/output/tensors_mrnh19657', f.frozen_tensors_dir
assert f.train_cache_dir == '/home/ict2/Projects/scDFM/cache/cache_mrnh19657', f.train_cache_dir
assert f.ae_encoder_ckpt.endswith('autoencoder_mrnh19657/encoder_2026-10-07_16-04.pt'), f.ae_encoder_ckpt
assert f.ae_decoder_ckpt.endswith('autoencoder_mrnh19657/decoder_2026-10-07_16-04.pt'), f.ae_decoder_ckpt
assert f.ae_latent_dim == 2048, f.ae_latent_dim
assert f.resolve_ntoken() == 2048 + 4, f.resolve_ntoken()  # AE 模式=潜词表
assert f.resolve_pert_ntoken() == f.ntoken, f.resolve_pert_ntoken()  # 扰动表独立
assert f.panel_csv_path.endswith('mrnh19657/pert_counts.csv'), f.panel_csv_path
assert (f.ntoken, f.n_top_genes, f.infer_top_gene) == (19661, 19657, 19657)
assert f.inference_control_paths == [], f.inference_control_paths  # 2026-10-06 拆段：control 源在 inference 节
assert f.cache_workers == 18
assert f.split_method == 'whole' and f.use_mmd_loss is False
print('[1] FlowConfig 全部值来自 YAML OK')

# 2) CLI 覆盖 YAML
sys.argv = ['test', '--steps', '123', '--use-mmd-loss']
f2 = ConfigUtils.load(FlowConfig)
assert f2.steps == 123, f2.steps
assert f2.use_mmd_loss is True, f2.use_mmd_loss
assert f2.batch_size_per_gpu == 32, f2.batch_size_per_gpu
print('[2] CLI 覆盖 YAML OK')

# 3) 子类：common + flow 生效，子类重定义字段（batch_size_per_gpu=3）不被 YAML 覆盖
sys.argv = ['test']
g = ConfigUtils.load(GenConfig)
assert g.train_set_paths == f.train_set_paths, g.train_set_paths
assert g.ntoken == 19661 and g.d_model == 512, (g.ntoken, g.d_model)
assert g.batch_size_per_gpu == 3, g.batch_size_per_gpu
assert g.panel_csv_path.endswith('mrnh19657/pert_counts.csv')
print('[3] GenConfig: common+flow 生效、batch_size_per_gpu 保持子类默认 OK')

# 4) 无默认值验证：YAML 缺 key → 直接报错
import tempfile
import yaml
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    flow_src = yaml.safe_load(open('configs/universal.yaml'))['train']  # 2026-10-06 flow→train
    yaml.safe_dump({'train': {k: v for k, v in flow_src.items() if k != 'steps'}}, tf)
    bad = tf.name
try:
    ConfigUtils.load(FlowConfig, use_cli=False, yaml_path=bad)
    raise AssertionError('缺 key 未报错')
except TypeError as e:
    print('[4] YAML 缺 key 报错 OK:', e)
os.unlink(bad)

# 5) 派生路径（新语料缓存尚未构建：打印派生键，容忍 FileNotFoundError）
try:
    cache, mask, vocab = artifact_paths(g)
    print(f'[5] cache={cache}')
    print(f'[5] mask ={mask}')
    print(f'[5] vocab={vocab}')
except FileNotFoundError as e:
    print('[5] 缓存未构建（预期）:', str(e).splitlines()[0])
print('[5] 派生键 OK（缓存构建后才会命中）')

# 6) 阶段段拆分验证（2026-10-06：common 节删除，network+train 自动合并，
# inference/build_* 段覆盖）
sys.argv = ['test']
c = ConfigUtils.load(FlowConfig, use_cli=False)  # network + train
assert c.frozen_tensors_dir == '/home/ict2/Projects/scDFM/output/tensors_mrnh19657'
assert c.perturb_direction == ['CRISPRi'], c.perturb_direction
assert (c.n_ctrl_cells, c.n_pred_cells) == (0, 0), (c.n_ctrl_cells, c.n_pred_cells)  # 训练入口默认空
assert c.fusion_method == 'differential_transformer' and c.ae_latent_dim == 2048, 'network 段未合并'
b = ConfigUtils.load(FlowConfig, use_cli=False, extra_sections=('build_tensors',))
assert (b.n_ctrl_cells, b.n_pred_cells, b.n_real_cells, b.min_real_cells) == (18400, 400, 2000, 100)
print('[6] 阶段段拆分（network+train 合并、build_tensors 覆盖）OK')

# 7) derive_pert_columns 直查口径（2026-10-07：mrnh19657 统一列名后删候选机制）
import pandas as pd
import numpy as np
from src.utils.utils import derive_pert_columns
o1 = pd.DataFrame({'target_gene': ['BRCA1', 'non-targeting', 'TP53', 'ctrl'],
                   'context': ['K562'] * 4})
d1 = derive_pert_columns(o1)
assert list(d1['target_gene']) == ['BRCA1', 'non-targeting', 'TP53', 'non-targeting']
assert list(d1['context']) == ['K562'] * 4
# 缺列 → 报错
try:
    derive_pert_columns(pd.DataFrame({'x': [1]}))
    raise AssertionError('缺列未报错')
except KeyError:
    pass
print('[7] derive_pert_columns 直查 + 哨兵规范化 + 缺列报错 OK')
print('ALL PASS')

# [8] bool flag 门控：下划线/连字符形式等价（2026-09-29 修复 worker --reuse_real 失效）
import dataclasses as _dc
from src.utils.config_utils import ConfigUtils as _CU
@_dc.dataclass
class _BoolCfg:
    reuse_real: bool = False
    batch_size: int = 0
_sys = __import__('sys')
for _form in ('--reuse_real', '--reuse-real'):
    _sys.argv = ['x', _form]
    _c = _BoolCfg()
    _CU._apply_cli(_c, _BoolCfg, None)
    assert _c.reuse_real is True, _form
_sys.argv = ['x', '--no-reuse-real']
_c = _BoolCfg()
_CU._apply_cli(_c, _BoolCfg, None)
assert _c.reuse_real is False
print('[8] bool flag 下划线/连字符等价 OK')
