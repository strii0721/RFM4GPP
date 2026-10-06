import os
import sys

from src.utils.config_utils import CommonConfig, ConfigUtils, FlowConfig
from tmp.generate_submission import GenConfig, artifact_paths

# 1) 无 CLI 参数：全部值来自 YAML（dataclass 无默认值）
sys.argv = ['test']
f = ConfigUtils.load(FlowConfig)
assert f.batch_size == 96, f.batch_size
assert f.steps == 100000, f.steps
assert f.early_stop_patience == 10, f.early_stop_patience
assert f.resume == '', f.resume
assert f.ntoken == 19551, f.ntoken
assert len(f.train_set_paths) == 18, len(f.train_set_paths)
assert f.train_set_paths[0].endswith('mrnh19547/marson_d1_rest.h5ad')
assert f.train_set_paths[-1].endswith('mrnh19547/h1.h5ad'), f.train_set_paths[-1]
assert f.perturb_direction == ['CRISPRi'], f.perturb_direction
assert f.obs_col_candidates == {
    'pert_gene': ['target_gene', 'perturbation'],
    'ctrl_flag': ['is_control'],
    'cell_line': ['context', 'cell_line_name'],
}, f.obs_col_candidates
assert f.ctrl_sentinels == ['non-targeting', 'non_targeting', 'ctrl', 'control', 'ntc', 'negative_control'], f.ctrl_sentinels
assert f.output_base_dir == 'output', f.output_base_dir
assert f.log_base_dir == 'logs', f.log_base_dir
assert f.make_path('train').startswith('output/train_'), f.make_path('train')
assert f.make_path('inference').startswith('output/inference_'), f.make_path('inference')
assert (f.n_ctrl_cells, f.n_pred_cells, f.n_real_cells, f.min_real_cells) == (18400, 400, 2000, 100)
assert f.test_set_paths == '/ssd3/PubData/gene_alignment/Replogle_RPE1/Replogle_RPE1.h5ad', f.test_set_paths
assert f.frozen_tensors_dir == '/home/ict2/Projects/scDFM/output/frozen_tensors_mrnh19547', f.frozen_tensors_dir
assert f.train_cache_dir == '/home/ict2/Projects/scDFM/cache/train_cache_mrnh19547', f.train_cache_dir
assert f.ae_encoder_ckpt.endswith('encoder_2026-10-06_12-34.pt'), f.ae_encoder_ckpt  # AE warm-start（2026-10-06）
assert f.ae_decoder_ckpt.endswith('decoder_2026-10-06_12-34.pt'), f.ae_decoder_ckpt
assert f.ae_latent_dim == 4096, f.ae_latent_dim
assert f.resolve_ntoken() == 4096 + 4, f.resolve_ntoken()  # AE 模式=潜词表
assert f.resolve_pert_ntoken() == f.ntoken, f.resolve_pert_ntoken()  # 扰动表独立
assert f.panel_csv_path.endswith('controls/pert_counts.csv'), f.panel_csv_path
assert (f.ntoken, f.n_top_genes, f.infer_top_gene) == (19551, 19547, 19547)
assert len(f.inference_control_paths) == 3
assert all(p.endswith(f'controls/context_{c}.h5ad')
           for p, c in zip(f.inference_control_paths, 'ABC')), f.inference_control_paths
assert f.cache_workers == 18
assert f.split_method == 'whole' and f.use_mmd_loss is False
print('[1] FlowConfig 全部值来自 YAML OK')

# 2) CLI 覆盖 YAML
sys.argv = ['test', '--steps', '123', '--use-mmd-loss']
f2 = ConfigUtils.load(FlowConfig)
assert f2.steps == 123, f2.steps
assert f2.use_mmd_loss is True, f2.use_mmd_loss
assert f2.batch_size == 96, f2.batch_size
print('[2] CLI 覆盖 YAML OK')

# 3) 子类：common + flow 生效，子类重定义字段（batch_size=3）不被 YAML 覆盖
sys.argv = ['test']
g = ConfigUtils.load(GenConfig)
assert g.train_set_paths == f.train_set_paths, g.train_set_paths
assert g.ntoken == 19551 and g.d_model == 512, (g.ntoken, g.d_model)
assert g.batch_size == 3, g.batch_size
assert g.panel_csv_path.endswith('controls/pert_counts.csv')
print('[3] GenConfig: common+flow 生效、batch_size 保持子类默认 OK')

# 4) 无默认值验证：YAML 缺 key → 直接报错
import tempfile
import yaml
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    flow_src = yaml.safe_load(open('configs/universal.yaml'))['flow']
    yaml.safe_dump({'flow': {k: v for k, v in flow_src.items() if k != 'steps'}}, tf)
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

# 6) CommonConfig 纯 YAML
sys.argv = ['test']
c = ConfigUtils.load(CommonConfig, use_cli=False)
assert c.frozen_tensors_dir == '/home/ict2/Projects/scDFM/output/frozen_tensors_mrnh19547'
assert c.test_set_paths.endswith('Replogle_RPE1.h5ad'), c.test_set_paths
assert c.perturb_direction == ['CRISPRi'], c.perturb_direction
assert c.obs_col_candidates['pert_gene'] == ['target_gene', 'perturbation'], c.obs_col_candidates
assert c.output_base_dir == 'output' and c.log_base_dir == 'logs', (c.output_base_dir, c.log_base_dir)
assert (c.n_ctrl_cells, c.n_pred_cells, c.n_real_cells, c.min_real_cells) == (18400, 400, 2000, 100)
print('[6] CommonConfig 纯 YAML OK')

# 7) derive_pert_columns 候选列匹配（合成三套 schema：旧式/新式/第三命名方）
import pandas as pd
import numpy as np
from src.utils.utils import derive_pert_columns
cand = c.obs_col_candidates
sent = c.ctrl_sentinels
# 旧式：target_gene + context + is_control
o1 = pd.DataFrame({'target_gene': ['BRCA1', 'non-targeting', 'TP53'],
                   'context': ['K562', 'K562', 'K562'],
                   'is_control': [False, True, False]})
d1 = derive_pert_columns(o1, cand, sent)
assert list(d1['target_gene']) == ['BRCA1', 'non-targeting', 'TP53']
assert list(d1['context']) == ['K562', 'K562', 'K562']
# 新式：perturbation + cell_line_name + is_control
o2 = pd.DataFrame({'perturbation': ['BRCA1', 'ctrl', 'TP53'],
                   'cell_line_name': ['Jurkat'] * 3,
                   'is_control': [False, True, False]})
d2 = derive_pert_columns(o2, cand, sent)
assert list(d2['target_gene']) == ['BRCA1', 'non-targeting', 'TP53']
assert list(d2['context']) == ['Jurkat'] * 3
# 第三命名方：gene 列叫 knocked_gene、系叫 line、无对照标记列（用哨兵值判对照）
o3 = pd.DataFrame({'knocked_gene': ['BRCA1', 'CTRL', 'TP53'],
                   'line': ['HepG2'] * 3})
cand3 = {'pert_gene': ['knocked_gene'], 'cell_line': ['line'], 'ctrl_flag': []}
d3 = derive_pert_columns(o3, cand3, ['ctrl', 'non-targeting'])
assert list(d3['target_gene']) == ['BRCA1', 'non-targeting', 'TP53']
assert list(d3['context']) == ['HepG2'] * 3
# 全部候选未命中 → 报错
try:
    derive_pert_columns(pd.DataFrame({'x': [1]}), cand, sent)
    raise AssertionError('候选未命中未报错')
except KeyError:
    pass
print('[7] derive_pert_columns 候选匹配 OK（旧式/新式/第三命名方/未命中报错）')
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
