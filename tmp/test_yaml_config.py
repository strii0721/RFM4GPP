import os
import sys

from src.utils.config_utils import CommonConfig, ConfigUtils, FlowConfig
from tmp.generate_submission import GenConfig, artifact_paths

# 1) 无 CLI 参数：全部值来自 YAML（dataclass 无默认值）
sys.argv = ['test']
f = ConfigUtils.load(FlowConfig)
assert f.batch_size == 96, f.batch_size
assert f.steps == 10000, f.steps
assert f.ntoken == 18537, f.ntoken
assert f.train_set_paths == [
    '/ssd3/PubData/gene_alignment/Replogle_K562_gwps/Replogle_K562_gwps.h5ad',
    '/ssd3/PubData/gene_alignment/Replogle_K562_essential/Replogle_K562_essential.h5ad',
    '/ssd3/PubData/gene_alignment/Nadig_Jurkat/Nadig_Jurkat.h5ad',
    '/ssd3/PubData/gene_alignment/Nadig_HepG2/Nadig_HepG2.h5ad',
], f.train_set_paths
assert f.test_set_paths == '/ssd3/PubData/gene_alignment/Replogle_RPE1/Replogle_RPE1.h5ad', f.test_set_paths
assert f.frozen_tensors_dir == '/home/ict2/Projects/scDFM/output/frozen_tensors_bad_aligned_replogle', f.frozen_tensors_dir
assert f.panel_csv_path.endswith('bad_aligned_replogle/pert_counts.csv'), f.panel_csv_path
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
assert g.ntoken == 18537 and g.d_model == 512, (g.ntoken, g.d_model)
assert g.batch_size == 3, g.batch_size
assert g.panel_csv_path.endswith('bad_aligned_replogle/pert_counts.csv')
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
assert c.frozen_tensors_dir == '/home/ict2/Projects/scDFM/output/frozen_tensors_bad_aligned_replogle'
assert c.test_set_paths.endswith('Replogle_RPE1.h5ad'), c.test_set_paths
print('[6] CommonConfig 纯 YAML OK')
print('ALL PASS')
