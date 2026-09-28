import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import json
import torch
import pickle
from typing import Union, Optional
from pathlib import Path
import os
from src.utils._preprocessing import annotate_compounds, get_molecular_fingerprints
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pdb
import tqdm
from random import shuffle
from scipy import sparse
from src.utils.utils import build_gene_coexpression_graph,sorted_pad_mask, derive_pert_columns
# combosciplex url: https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635
# 'norman' url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'

class Data:
    def __init__(self, data_path='../../data', config=None):
        self.data_path = data_path
        self.config = config
        # data_path 可为相对路径（如 'cache'，须在项目根运行）——自动创建
        os.makedirs(data_path, exist_ok=True)

        
    def load_data(self, data_name = None, data_path = None):
        self.data_name = data_name
        if data_name in ['norman', 'norman_umi_go_filtered',]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        elif data_name in ['combosciplex', ]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        elif data_name == 'vcc':
            # 2026-09-17 内存优化：缓存存在时直接读缓存、跳过 19GB 语料解压读入
            # （语料仅用于建缓存；缓存就绪后训练/预构建不再需要它）。
            # process_data 的 vcc 分支凭 _loaded_from_cache 标记不再重复读。
            pool_stem = (os.path.splitext(os.path.basename(str(self.config.train_pool_path)))[0]
                         if self.config.train_pool_path else 'all')
            # 多文件语料（2026-09-27）：缓存键 stem = 各文件名（去扩展名）排序后 + 拼接，
            # 与 generate_submission.corpus_stem / process_data 的派生规则一致
            corpus_stem = '+'.join(os.path.splitext(os.path.basename(str(p)))[0]
                                   for p in sorted(self.config.train_set_paths))
            cache = os.path.join(self.config.train_cache_dir,
                                 f'processed_n{self.config.n_top_genes}_'
                                 f'{corpus_stem}'
                                 f'_{pool_stem}.h5ad')
            if os.path.exists(cache):
                self._loaded_from_cache = True
                # 共享内存加载（2026-09-21）：X 侧车 .npy 存在时直接读 meta（MB 级）
                # + memmap 零拷贝映射，**跳过整 h5ad 读取**（曾每 rank 白读 59GB、
                # 白分配 62GB 私有内存；read_bytes 实测 65GB/rank）。多 rank 共享同一
                # 份文件页缓存（~62GB）而非各持私有拷贝——共享机他人作业挤占内存时
                # 私有拷贝会被内核 OOM（当日三次 SIGKILL）。文件页可回收，memmap
                # 版本基本免疫 OOM killer。
                sidecar = cache + '.data.npy'
                if os.path.exists(sidecar):
                    self.adata = sc.read_h5ad(cache + '.meta.h5ad')
                    d = np.load(cache + '.data.npy', mmap_mode='r')
                    idx = np.load(cache + '.indices.npy', mmap_mode='r')
                    ptr = np.load(cache + '.indptr.npy', mmap_mode='r')
                    # 注意：csr_matrix((d,idx,ptr), copy=False) 仍走 COO→CSR 转换并
                    # 无条件拷贝（实测 RSS 92GB）；零拷贝必须空构造 + 属性直赋。
                    X = sparse.csr_matrix(self.adata.shape, dtype=d.dtype)
                    X.data = d
                    X.indices = idx
                    X.indptr = ptr
                    # 关键（2026-09-21 两次训练崩因）：属性直赋后 scipy 不缓存
                    # canonical 状态，首次切片会触发 sum_duplicates 全量校验拷贝
                    # （实测 38s + 92GB/进程；8 rank × 4 worker 并发 → cgroup
                    # 768GB 限额秒爆，worker 被 OOM kill）。数组源自 anndata 写出
                    # 的规范 csr（行内已排序、无重复，经逐位验证），直接置真。
                    X.has_sorted_indices = True
                    X.has_canonical_format = True
                    self.adata.X = X
                    print(f'##### load_data: X via shared memmap sidecars: {sidecar} #####')
                else:
                    self.adata = sc.read_h5ad(cache)
                print(f'##### load_data: cache hit, corpus read skipped: {cache} #####')
            else:
                # 多文件语料（2026-09-27 用户定案）：按序全量读入、沿 obs 拼接。
                # var 轴与 obs 列集合逐文件校验一致，不一致直接报错（缓存派生键已含各文件名）。
                paths = list(self.config.train_set_paths)
                assert paths, 'train_set_paths 为空'
                parts = []
                var0 = None
                for p in paths:
                    a = sc.read_h5ad(p)
                    a.obs = derive_pert_columns(a.obs)  # 新对齐语料：perturbation/is_control→target_gene、cell_line_name→context
                    if var0 is None:
                        var0 = list(a.var_names)
                    else:
                        assert list(a.var_names) == var0, f'var 轴不一致: {p}'
                    parts.append(a)
                    print(f'##### load_data: corpus part {p}: {a.shape} #####')
                self.adata = ad.concat(parts, join='outer', index_unique=None)
                print(f'##### load_data: corpus concat {len(parts)} files -> {self.adata.shape} #####')
                self._loaded_from_cache = False
        else:
            raise ValueError(data_name + ' is not a valid data name')
        
    def process_data(self, n_top_genes = 2000,infer_top_gene=1000,split_method='additive',
                     use_negative_edge=True, k=30,
                     **kwargs):
        os.makedirs(os.path.join(self.data_path, self.data_name), exist_ok=True)
        if self.data_name == 'combosciplex':
            
            if os.path.exists(os.path.join(self.data_path, self.data_name, 'processed.h5ad')):
                self.adata = sc.read_h5ad(os.path.join(self.data_path, self.data_name, 'processed.h5ad'))
            else:   

                self.adata.obs["condition"] = self.adata.obs.apply(
                    lambda x: "control" if x["condition"] == "control+control" else x["condition"], axis=1
                )

                self.adata.obs["is_control"] = self.adata.obs.apply(
                    lambda x: True if x["condition"] == "control" else False, axis=1
                )
                
                annotate_compounds(self.adata, compound_keys=["Drug1", "Drug2"])
                get_molecular_fingerprints(self.adata, compound_keys=["Drug1", "Drug2"])
                self.adata.uns["fingerprints"]["control"] = np.zeros(1024)
                
                self.adata.write(os.path.join(self.data_path, self.data_name, 'processed.h5ad'))
            
            self.adata.X = self.adata.layers["counts"].copy()
            sc.pp.normalize_total(self.adata)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, inplace=True, n_top_genes=n_top_genes)
                
            if 'test_conditions' in kwargs.keys():
                test_conditions = kwargs['test_conditions']
            else:
                test_conditions = ['Panobinostat+Crizotinib', 
                                'Panobinostat+Curcumin', 
                                'Panobinostat+SRT1720', 
                                'Panobinostat+Sorafenib', 
                                'SRT2104+Alvespimycin', 
                                'control+Alvespimycin', 
                                'control+Dacinostat']
                
            self.adata = self.adata[:,self.adata.var['highly_variable']] # filter out low variable genes
            
            self.adata.obs["mode"] = self.adata.obs.apply(lambda x: "test" if x["condition"] in test_conditions else "train", axis=1)
            self.adata_train = self.adata[self.adata.obs["mode"] == "train"]
            self.adata_test = self.adata[(self.adata.obs["mode"] == "test") | (self.adata.obs["condition"]=="control")]
            
            sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
            self.adata_test = self.adata_test[:,self.adata_test.var['highly_variable']]
            
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            # self._val_manager = 
        elif self.data_name == 'norman' or self.data_name == 'norman_umi_go_filtered':
            
            sc.pp.highly_variable_genes(self.adata, inplace=True, n_top_genes=n_top_genes)
            unique_perturbation = []
            [unique_perturbation.extend(perturbation.split('+')) for perturbation in self.adata.obs['condition'].unique()]
            unique_perturbation = np.unique(unique_perturbation)
            
            if self.data_name == 'norman':
                for perturbation in unique_perturbation:
                    if perturbation in self.adata.var_names:
                        self.adata.var.loc[perturbation, 'highly_variable'] = True
                    else:
                        print(f"Warning: {perturbation} is not in the gene names")
                self.adata = self.adata[:,self.adata.var['highly_variable']]
            elif self.data_name == 'norman_umi_go_filtered':
                all_gene_names = list(self.adata.var['gene_name']) + ['ctrl']
                for perturbation in unique_perturbation:
                    if perturbation not in all_gene_names:
                        print(f"Warning: {perturbation} is not in the gene names")
                self.adata.var['highly_variable'] = True

            
            #### for split five times 
            if split_method == 'additive' or 'combinations':
                split_file = os.path.join(self.data_path, self.data_name, 'split_results.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    perturbations = np.unique(self.adata.obs['condition'])
                    double_perturbation = [p for p in perturbations if 'ctrl' not in p]
                    double_perturbation = np.array(double_perturbation)

                    self.split_results = []
                    
                    for i in range(5):
                        np.random.seed(42 + i)
                        shuffled = double_perturbation.copy()
                        np.random.shuffle(shuffled)
                        
                        split_idx = int(len(shuffled) * 0.3)
                        test_double = shuffled[:split_idx]
                        train_double = shuffled[split_idx:]
                        self.split_results.append({
                            'train': train_double.tolist(),
                            'test': test_double.tolist()
                        })
                    
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results saved')
                    
            elif split_method == 'unseen':
                split_file = os.path.join(self.data_path, self.data_name, 'split_results_unseen.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    self.split_results = []
                    for i in range(5):
                        perturbations = np.unique(self.adata.obs['condition'])
                        double_perturbation = [p for p in perturbations if 'ctrl' not in p]
                        single = []
                        [single.extend(p.split('+')) for p in double_perturbation]
                        single = list(set(single))
                    
                        shuffle(single)
                        remove_genes = single[:12]
                        p_count = {}
                        for p in double_perturbation:
                            ps = p.split('+')
                            count = int(ps[0] in remove_genes) + int(ps[1] in remove_genes)
                            p_count[p] = count
                        double_perturbation = [p for p, count in p_count.items() if count > 0]
                        double_perturbation = list(double_perturbation)
                        remove_genes_condition = [p+'+control' for p in remove_genes]
                        double_perturbation.extend(remove_genes_condition)
                        self.split_results.append({
                            'p_count': p_count,
                            'test': double_perturbation
                        })
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results unseen saved')
            
            if 'fold' in kwargs.keys():
                fold = kwargs['fold']
            else:
                fold = 0
            self.adata.obs['condition'] = self.adata.obs['condition'].str.replace('ctrl', 'control')
            self.adata.obs['Drug1'] = self.adata.obs['condition'].str.split('+').apply(lambda x: x[0])
            self.adata.obs['Drug2'] = self.adata.obs['condition'].str.split('+').apply(lambda x: x[-1])
            self.adata.obs['is_control'] = False
            self.adata.obs.loc[self.adata.obs['control'] == 1, 'is_control'] = True
            self.adata.obs['mode'] = 'train'
            
            
            if split_method == 'combinations':
                self.split_results[fold]['test'] = self.split_results[fold]['test'][:15]
                remove_genes = []
                [remove_genes.extend(p.split('+')) for p in self.split_results[fold]['test']]
                remove_genes = set(remove_genes)
                remove_genes_condition = [p+'+control' for p in remove_genes]
                
                self.split_results[fold]['test'].extend(remove_genes_condition)            
            
            
            self.adata.obs.loc[self.adata.obs['condition'].isin(self.split_results[fold]['test']), 'mode'] = 'test'
            
            self.adata_train = self.adata[self.adata.obs['mode'] == 'train']
            self.adata_test = self.adata[(self.adata.obs['mode'] == 'test') | (self.adata.obs['control'] == 1)]
            
            
            
            sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
            self.adata_test = self.adata_test[:,self.adata_test.var['highly_variable']]
            
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            
        elif self.data_name == 'vcc':
            cfg = self.config
            assert cfg is not None, 'vcc mode requires Data(config=...)'
            corpus_stem = '+'.join(os.path.splitext(os.path.basename(str(p)))[0]
                                   for p in sorted(cfg.train_set_paths))
            # 缓存键含采样池（2026-09-17：缓存列= train_pool_path 清单；换池须重建）
            pool_stem = (os.path.splitext(os.path.basename(str(cfg.train_pool_path)))[0]
                         if cfg.train_pool_path else 'all')
            cache = os.path.join(cfg.train_cache_dir,
                                 f'processed_n{n_top_genes}_{corpus_stem}_{pool_stem}.h5ad')
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            if os.path.exists(cache):
                if getattr(self, '_loaded_from_cache', False):
                    print(f'##### processed h5ad already loaded from cache: {cache} #####')
                else:
                    print(f'##### loading cached processed h5ad: {cache} #####')
                    self.adata = sc.read_h5ad(cache)
            else:
                # 1) CRISPRi only (drop CRISPR KO)
                if cfg.crispr_type_col and cfg.crispr_type_col in self.adata.obs:
                    keep = self.adata.obs[cfg.crispr_type_col].astype(str) == cfg.crispr_type_value
                    print(f'##### vcc: keeping {keep.sum()}/{self.adata.n_obs} {cfg.crispr_type_value} cells #####')
                    self.adata = self.adata[keep].copy()
                # 2) condition / is_control from target_gene (single 'non-targeting' label)
                tg = self.adata.obs['target_gene'].astype(str).to_numpy()
                is_ctl = tg == 'non-targeting'
                self.adata.obs['condition'] = np.where(is_ctl, 'control', tg + '+control')
                self.adata.obs['is_control'] = is_ctl
                # 3) 列过滤：train_pool_path 非空 = 清单（如 common_hvg）∩ 语料 var；
                #    空串（2026-09-19 定案）= 整个基因轴，先全轴预处理、第 5 步 HVG
                #    裁到 n_top_genes（11919 → 实质全部有 dispersion 的列）。
                if cfg.train_pool_path:
                    pool_raw = pd.read_csv(cfg.train_pool_path)['gene_name'].astype(str).tolist()
                    keep_cols = [g for g in pool_raw if g in set(self.adata.var_names)]
                    assert keep_cols, f'train_pool_path={cfg.train_pool_path!r} yields no usable genes'
                    self.adata = self.adata[:, keep_cols].copy()
                    print(f'##### vcc: cache columns filtered to {len(keep_cols)} genes '
                          f'from {cfg.train_pool_path} #####')
                else:
                    print(f'##### vcc: full-axis cache (train_pool_path empty, '
                          f'{self.adata.n_vars} genes before HVG) #####')
                # 4) paper preprocessing: normalize_total(CP10k) -> log1p（过滤后矩阵，快）
                #    (log-space linear paths, upstream combosciplex path)
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
                # 5) HVG：清单池时代 n_top=全部列只算 dispersions_norm（推理 ranking 用）
                #    不裁列；全轴时代 n_top=n_top_genes（11919）实质保留全部有
                #    dispersion 的列（零方差列 dispersion=NaN 永不入选）
                sc.pp.highly_variable_genes(
                    self.adata,
                    n_top_genes=(self.adata.n_vars if cfg.train_pool_path else n_top_genes))
                if not cfg.train_pool_path:
                    hv = self.adata.var['highly_variable'].to_numpy()
                    self.adata = self.adata[:, hv].copy()
                    print(f'##### vcc: full-axis cache kept {int(hv.sum())}/{len(hv)} '
                          f'genes (zero-variance dropped) #####')
                # obs/_index from the merged corpus reads back as a pandas
                # StringArray; anndata <0.13 refuses to write nullable strings
                # unless opted in (0.13+ default-on, setting may be removed)
                if hasattr(ad.settings, 'allow_write_nullable_strings'):
                    ad.settings.allow_write_nullable_strings = True
                # 缓存写出前把稀疏 X 的列索引 int64→int32：nnz 7.7e9 > 2^31 时
                # scipy 默认 int64 索引（每 rank RAM ~92GB）；列号 < 2^31 恒成立
                # （11,371 列），纯索引降精度、数值位不变，每 rank 降到 ~61GB。
                if hasattr(self.adata.X, 'indices'):
                    self.adata.X.indices = self.adata.X.indices.astype(np.int32, copy=False)
                self.adata.write(cache)
                # X 侧车（共享 memmap 加载用）：data/indices/indptr 三组 .npy + 无 X 的
                # meta.h5ad（obs/var/uns）。load_data 检测到侧车即走零拷贝共享加载。
                np.save(cache + '.data.npy', self.adata.X.data)
                np.save(cache + '.indices.npy', self.adata.X.indices)
                np.save(cache + '.indptr.npy', self.adata.X.indptr)
                sc.AnnData(X=None, obs=self.adata.obs, var=self.adata.var,
                           uns=self.adata.uns).write(cache + '.meta.h5ad')
                print(f'##### vcc: processed cached to {cache} #####')
            # 5) split：默认 single_line（HCT116 留系，2026-09-14 取代五折）；
            #    'single' = 80/20 panel 基因留出 x5 折（原方案，fold 选折）。
            #    held-out 基因/系 cells + all control cells form the test set,
            #    mirroring upstream's test|control pattern.
            if split_method == 'single':
                tg_all = self.adata.obs['target_gene'].astype(str)
                panel_genes = sorted(g for g in tg_all.unique() if g != 'non-targeting')
                # split file keyed by corpus (per-corpus gene sets differ)
                split_file = os.path.join(self.data_path, self.data_name,
                                          f'split_results_single_{corpus_stem}.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    rng = np.random.default_rng(0)
                    shuffled = np.array(panel_genes)[rng.permutation(len(panel_genes))]
                    self.split_results = []
                    n_test = max(1, int(round(len(panel_genes) * 0.2)))
                    for i in range(5):
                        test_genes = shuffled[i * n_test:(i + 1) * n_test].tolist()
                        self.split_results.append({
                            'train': [g for g in panel_genes if g not in set(test_genes)],
                            'test': test_genes,
                        })
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results saved')
                fold = kwargs.get('fold', 0)
                test_genes = set(self.split_results[fold]['test'])
                is_test = tg_all.isin(test_genes).to_numpy()
                self.adata.obs['mode'] = np.where(is_test, 'test', 'train')
                self.adata.obs['Drug1'] = self.adata.obs['condition'].str.split('+').str[0]
                self.adata.obs['Drug2'] = self.adata.obs['condition'].str.split('+').str[-1]
                self.adata_train = self.adata[self.adata.obs['mode'] == 'train'].copy()
                self.adata_test = self.adata[(self.adata.obs['mode'] == 'test') | self.adata.obs['is_control']].copy()
                n_ctl_test = int(self.adata_test.obs['is_control'].sum())
                print(f'##### vcc: fold {fold} train {self.adata_train.n_obs} cells / '
                      f'test {self.adata_test.n_obs} cells (held-out genes {len(test_genes)}, ctl {n_ctl_test}) #####')
                # test-set HVG selection (infer_top_gene), upstream-style
                sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
                self.adata_test = self.adata_test[:, self.adata_test.var['highly_variable']]
            elif split_method == 'single_line':
                # 留一系（2026-09-14）：train = 剔除 heldout_line 全部细胞；
                # test = 该系全部细胞（扰动 + 对照），作本地官方 benchmark 的
                # 真实参考。mask/vocab 都从 train 建（该系完全不可见）。
                line_vals = self.adata.obs[cfg.line_col].astype(str).to_numpy()
                is_held = line_vals == cfg.heldout_line
                n_held = int(is_held.sum())
                assert n_held > 0, f'heldout_line={cfg.heldout_line!r} not found in {cfg.line_col}'
                self.adata.obs['mode'] = np.where(is_held, 'test', 'train')
                self.adata.obs['Drug1'] = self.adata.obs['condition'].str.split('+').str[0]
                self.adata.obs['Drug2'] = self.adata.obs['condition'].str.split('+').str[-1]
                self.adata_train = self.adata[~is_held].copy()
                self.adata_test = self.adata[is_held].copy()
                print(f'##### vcc: single_line holdout {cfg.heldout_line}: train '
                      f'{self.adata_train.n_obs} cells / test {self.adata_test.n_obs} cells #####')
                # test-set HVG selection (infer_top_gene), upstream-style
                sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
                self.adata_test = self.adata_test[:, self.adata_test.var['highly_variable']]
            elif split_method == 'whole':
                # 全文件训练（replogle 2026-09-17）：语料文件本身无留出；
                # 测试语料 = 独立文件 config.test_set_paths，由
                # benchmark_line_holdout.py 直接读取（do_eval 必须 False）。
                # adata_test 给零行空壳，兼容 TestDataset 构造（不用其方法）。
                self.adata.obs['mode'] = 'train'
                self.adata.obs['Drug1'] = self.adata.obs['condition'].str.split('+').str[0]
                self.adata.obs['Drug2'] = self.adata.obs['condition'].str.split('+').str[-1]
                # 2026-09-17 内存优化：whole 无切片，直接共享引用（TrainSampler 只加
                # obs 列、无 X 变异；copy() 会让每 rank 多持有一份 ~93GB 矩阵，8 rank 共爆内存）
                self.adata_train = self.adata
                self.adata_test = self.adata[0:0].copy()
                print(f'##### vcc: whole-corpus split: train {self.adata_train.n_obs} cells '
                      f'(no internal holdout; test file: {cfg.test_set_paths}) #####')
            else:
                raise ValueError(f'vcc requires split_method="single"/"single_line"/"whole", '
                                 f'got {split_method!r}')
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            self.split_results = self.split_results if hasattr(self, 'split_results') else []
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
        
        cfg = self.config
        if 'fold' in kwargs.keys():
            fold = kwargs['fold']
        else:
            fold = 0
        if self.data_name == 'vcc' and cfg is not None:
            # per-corpus mask file (graph built from this corpus's own train data);
            # signed/unsigned graphs are different artifacts -> name must differ
            _stem = '+'.join(os.path.splitext(os.path.basename(str(p)))[0]
                             for p in sorted(cfg.train_set_paths))
            _pool = (os.path.splitext(os.path.basename(str(cfg.train_pool_path)))[0]
                     if cfg.train_pool_path else 'all')
            _neg = '_negative_edge' if use_negative_edge else ''
            mask_path = os.path.join(self.data_path, self.data_name,
                                     f'mask_fold_{fold}topk_{k}{split_method}{_neg}_{_stem}_{_pool}.pt')
        elif use_negative_edge:
            mask_path = os.path.join(self.data_path, self.data_name,'mask_fold_'+str(fold)+'topk_'+str(k)+split_method+'_negative_edge'+'.pt')
        else:
            mask_path = os.path.join(self.data_path, self.data_name,'mask_fold_'+str(fold)+'topk_'+str(k)+split_method+'.pt')
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path)
        else:
            if self.data_name == 'vcc' and cfg is not None and cfg.mask_subsample and self.adata_train.n_obs > cfg.mask_subsample:
                rng = np.random.default_rng(42)
                sub_idx = rng.choice(self.adata_train.n_obs, cfg.mask_subsample, replace=False)
                X = self.adata_train.X[sub_idx].toarray()
                print(f'##### vcc: mask built from {cfg.mask_subsample} subsampled train cells #####')
            else:
                X = self.adata_train.X.toarray()
            mask = build_gene_coexpression_graph(X,
                method="pearson",
                wgcna_beta=None,
                sparsify="topk",
                k=k,
                use_negative_edge=use_negative_edge)
            mask = sorted_pad_mask(mask, pad_size=4, gene_names=list(self.adata_train.var_names))
            torch.save(mask, mask_path)
            print('mask saved')
        self.mask_path = mask_path
        
        
    def load_flow_data(self, batch_size = 128):
        if self.data_name == 'combosciplex':
            train_sampler = TrainSampler(self.data_name, self.adata_train, ["Drug1", "Drug2"], self.perturbation_dict)
            test_sampler = TestDataset(self.data_name, self.adata_test, ["Drug1", "Drug2"], self.perturbation_dict)
            
            return train_sampler , test_sampler, []
        elif self.data_name == 'norman' or self.data_name == 'norman_umi_go_filtered' or self.data_name == 'vcc':
            line_col = None
            min_tgt_cells = 1
            if self.data_name == 'vcc' and self.config is not None:
                line_col = getattr(self.config, 'line_col', None)
                min_tgt_cells = getattr(self.config, 'min_tgt_cells', 1)
            train_sampler = TrainSampler(self.data_name, self.adata_train, ["Drug1", "Drug2"], self.perturbation_dict,
                                         line_col=line_col, min_tgt_cells=min_tgt_cells)
            test_sampler = TestDataset(self.data_name, self.adata_test, ["Drug1", "Drug2"], self.perturbation_dict,
                                       line_col=line_col)
            return train_sampler , test_sampler, []
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
            

    def pretrain_data(self, batch_size = 128):
        if self.data_name == 'combosciplex':
            
            self.pretrain_train_data = PretrainData(self.adata_train, self.perturbation_dict)
            self.pretrain_train_data_loader = DataLoader(self.pretrain_train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            self.pretrain_test_data = PretrainData(self.adata_test, self.perturbation_dict)
            self.pretrain_test_data_loader = DataLoader(self.pretrain_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
            return self.pretrain_train_data_loader, self.pretrain_test_data_loader
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
        
class TrainSampler:
    def __init__(self, data_name, adata: sc.AnnData, perturbation_covariates: list[str], perturbation_dict: dict,
                 line_col: Optional[str] = None, min_tgt_cells: int = 1):
        self.data_name = data_name
        self.adata = adata
        self.perturbation_covariates = perturbation_covariates
        self.adata.obs['perturbation_covariates'] = self.adata.obs[perturbation_covariates].apply(lambda x: '+'.join(x), axis=1)
        self._perturbation_covariates = adata.obs['perturbation_covariates'].unique()
        
        self._perturbation_covariates = self._perturbation_covariates[self._perturbation_covariates != 'control+control']
        
        self._perturbation_covariates.sort()
        self.perturbation_covariates_dict = {perturbation: i for i, perturbation in enumerate(self._perturbation_covariates)}
        
        perturbation_covariates_id = [adata.obs[perturbation_covariates[i]].apply(lambda x: perturbation_dict[x])
                                    for i in range(len(perturbation_covariates))]
        self.perturbation_covariates_id = np.array(perturbation_covariates_id).T
        
        
        self.cells_name = self.adata.obs_names
        
        # ---- line-aware pairing (same-cell-line source/target sampling) ----
        # With multi-cell-line corpora the upstream implementation pairs a control
        # cell from ANY line with a perturbed cell from ANY line (independent
        # sampling). Under CFM the optimal velocity field is then independent of
        # the control condition (c ⊥ x1 in the training distribution), so the
        # model never learns line-consistent perturbation effects. Restrict both
        # source and target pools to ONE cell line per batch.
        self.line_col = line_col
        self.min_tgt_cells = min_tgt_cells
        self._line_aware = bool(line_col) and line_col in self.adata.obs.columns
        self._ctl_pool = {}
        self._tgt_pool = {}
        self._eligible_lines = {}
        if self._line_aware:
            lines = self.adata.obs[line_col].astype(str).to_numpy()
            pc = self.adata.obs['perturbation_covariates'].astype(str).to_numpy()
            self._lines = np.unique(lines)
            for L in self._lines:
                self._ctl_pool[L] = np.nonzero(np.logical_and(lines == L, pc == 'control+control'))[0]
            # 2026-09-26 向量化：旧实现逐 (pert, line) 做全列字符串比较
            # （8,679×3×225 万 ≈ 580 亿次，实测每次训练启动 ~25 min 纯 CPU）
            # → 每系一次 stable argsort + 边界切分（O(n log n)），池内容完全等价、秒级。
            elig_map: dict[str, list] = {}
            for L in self._lines:
                mask_L = lines == L
                base = np.nonzero(mask_L)[0]
                pc_L = pc[mask_L]
                order = np.argsort(pc_L, kind='stable')
                spc = pc_L[order]
                bounds = np.nonzero(spc[1:] != spc[:-1])[0] + 1
                starts = np.concatenate(([0], bounds))
                ends = np.concatenate((bounds, [spc.size]))
                ctl_ok = len(self._ctl_pool[L]) > 0
                for s, e in zip(starts, ends):
                    pert = spc[s]
                    if pert == 'control+control':
                        continue
                    if (e - s) >= min_tgt_cells and ctl_ok:
                        self._tgt_pool[(pert, L)] = base[order[s:e]]
                        elig_map.setdefault(pert, []).append(L)
            for pert in list(self._perturbation_covariates):
                self._eligible_lines[pert] = elig_map.get(pert, [])
            dropped = [p for p in self._perturbation_covariates if not self._eligible_lines[p]]
            if dropped:
                print(f'##### TrainSampler: dropping {len(dropped)} perturbations with no eligible line '
                      f'(min_tgt_cells={min_tgt_cells}): {dropped} #####')
            self._perturbation_covariates = np.array([p for p in self._perturbation_covariates
                                                      if self._eligible_lines[p]])
            self._perturbation_covariates.sort()
        
    def get_batch(self, batch_size: int, same_perturbation: bool = True):
        if same_perturbation:
            # random sample a perturbation from self._perturbation_covariates.
            # the last one is control
            perturbation_idx = np.random.choice(len(self._perturbation_covariates), 1)[0]
            
            perturbation_id = self._perturbation_covariates[perturbation_idx]
            
            if self._line_aware:
                # same-line pairing: sample one eligible cell line uniformly, then
                # draw both source (control) and target (perturbed) cells from that
                # line only
                elig = self._eligible_lines[perturbation_id]
                line = elig[np.random.randint(len(elig))]
                tgt_idx = self._tgt_pool[(perturbation_id, line)]
                src_idx = self._ctl_pool[line]
            else:
                tgt_idx = (self.adata.obs['perturbation_covariates'] == perturbation_id).to_numpy().nonzero()[0]
                src_idx = (self.adata.obs['perturbation_covariates'] == 'control+control').to_numpy().nonzero()[0]
            tgt_batch_idx = np.random.choice(tgt_idx, batch_size)
            src_batch_idx = np.random.choice(src_idx, batch_size)
            
            tgt_batch = torch.from_numpy(_row_slice_csr(self.adata.X, tgt_batch_idx).toarray())
            
            src_batch = torch.from_numpy(_row_slice_csr(self.adata.X, src_batch_idx).toarray())
            
            return {
                'src_cell_data': src_batch,
                'tgt_cell_data': tgt_batch,
                'src_cell_id': self.cells_name[src_batch_idx],
                'tgt_cell_id': self.cells_name[tgt_batch_idx],
                'condition_id': self.perturbation_covariates_id[tgt_batch_idx],
            }
            
        else:
            raise ValueError('same_perturbation must be True')
            
class TestDataset:
    def __init__(self, data_name,adata: sc.AnnData, perturbation_covariates: list[str], perturbation_dict: dict,
                 line_col: Optional[str] = None):
        self.data_name = data_name
        self.adata = adata
        self.perturbation_covariates = perturbation_covariates
        # 空 DataFrame（whole 切分的零行测试空壳）上 apply(axis=1) 返回多列 DataFrame
        # 而非 Series，赋值单列会 ValueError（2026-09-17 实测）——空表给空 Series。
        # ⚠️ dtype 必须 object：str 扩展类型（StringDtype）的 unique() 返回 StringArray，
        # 无 .sort() 方法，下一行会 AttributeError；非空路径 apply 产物也是 object。
        pc = self.adata.obs[perturbation_covariates]
        if pc.shape[0] == 0:
            self.adata.obs['perturbation_covariates'] = pd.Series(index=pc.index, dtype=object)
        else:
            self.adata.obs['perturbation_covariates'] = pc.apply(lambda x: '+'.join(x), axis=1)
        self._perturbation_covariates = adata.obs['perturbation_covariates'].unique()
        
        self._perturbation_covariates = self._perturbation_covariates[self._perturbation_covariates != 'control+control']
        
        self._perturbation_covariates.sort()
        self.perturbation_covariates_dict = {perturbation: i for i, perturbation in enumerate(self._perturbation_covariates)}
        
        perturbation_covariates_id = [adata.obs[perturbation_covariates[i]].apply(lambda x: perturbation_dict[x])
                                    for i in range(len(perturbation_covariates))]
        self.perturbation_covariates_id = np.array(perturbation_covariates_id).T
        
        
        self.cells_name = self.adata.obs_names
        
        # line-aware eval support: score each (perturbation, cell line) within its
        # own line's control/target pools instead of mixing all lines
        self.line_col = line_col
        self._line_aware = bool(line_col) and line_col in self.adata.obs.columns
        self._lines = np.unique(self.adata.obs[line_col].astype(str)) if self._line_aware else []
        
    def _mask(self, is_control: bool, line: Optional[str] = None, perturbation: Optional[str] = None):
        mask = self.adata.obs['is_control'].to_numpy() if is_control else \
            (self.adata.obs['perturbation_covariates'] == perturbation).to_numpy()
        if line is not None and self._line_aware:
            mask = mask & (self.adata.obs[self.line_col].astype(str).to_numpy() == line)
        return mask
    
    def get_control_data(self, line: Optional[str] = None):
        mask = self._mask(True, line=line)
        rows = np.nonzero(mask)[0]
        return {
            'src_cell_data': torch.from_numpy(_row_slice_csr(self.adata.X, rows).toarray()),
            'src_cell_id': self.adata.obs_names[rows],
            'condition_id': torch.tensor(self.perturbation_covariates_id[rows]),
        }

    def get_perturbation_data(self, perturbation: str, line: Optional[str] = None):
        mask = self._mask(False, line=line, perturbation=perturbation)
        rows = np.nonzero(mask)[0]
        return {
            'tgt_cell_data': torch.from_numpy(_row_slice_csr(self.adata.X, rows).toarray()),
            'tgt_cell_id': self.adata.obs_names[rows],
            'condition_id': torch.tensor(self.perturbation_covariates_id[rows]),
        }
    
    def perturbation_line_pairs(self, min_tgt_cells: int = 1):
        """(perturbation, line) combos to evaluate line-consistently."""
        pairs = []
        if self._line_aware:
            pc = self.adata.obs['perturbation_covariates'].astype(str).to_numpy()
            lines = self.adata.obs[self.line_col].astype(str).to_numpy()
            for pert in self._perturbation_covariates:
                for L in self._lines:
                    n = int(np.logical_and(pc == pert, lines == L).sum())
                    if n >= min_tgt_cells:
                        pairs.append((pert, L))
        else:
            pairs = [(p, None) for p in self._perturbation_covariates]
        return pairs
        
    
    
def _row_slice_csr(X, rows):
    """按行提取 csr 子矩阵，只读选中行。

    绕过 scipy 的 __getitem__/astype 路径：scipy 切片前会对 indices/indptr 调
    .astype(idx_dtype, copy=False)，对 memmap 支撑的数组 numpy 判定必须拷贝，
    于是每次切片物化整个 ~29GB indices（2026-09-21 cProfile 实锤：43s/次，
    32 worker 首批 ≈1TB anon → cgroup 768GB OOM 元凶）。本实现零全表扫描。"""
    rows = np.asarray(rows)
    starts = X.indptr[rows]
    ends = X.indptr[rows + 1]
    lens = (ends - starts).astype(np.int64, copy=False)
    total = int(lens.sum())
    out_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lens, out=out_indptr[1:])
    out_indices = np.empty(total, dtype=X.indices.dtype)
    out_data = np.empty(total, dtype=X.dtype)
    pos = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        n = e - s
        out_data[pos:pos + n] = X.data[s:e]
        out_indices[pos:pos + n] = X.indices[s:e]
        pos += n
    return sparse.csr_matrix((out_data, out_indices, out_indptr),
                             shape=(len(rows), X.shape[1]))


class PerturbationDataset(Dataset):
    def __init__(self, sampler: TrainSampler, batch_size: int, residual_dir: str = ''):
        self.sampler = sampler
        self.batch_size = batch_size
        self.perturbations = sampler._perturbation_covariates

        self.control_idx = (sampler.adata.obs['perturbation_covariates'] == 'control+control').to_numpy().nonzero()[0]

        # 残差目标范式（2026-09-26）：(line, pert) -> res_<line>.npy 行号；
        # combos.csv 行序按 line 块（同 build_residual_targets.py 落盘序）。
        self._combo_row_of = {}
        self._lines_list = []
        if residual_dir:
            combos = pd.read_csv(os.path.join(residual_dir, 'combos.csv'))
            self._lines_list = sorted(combos['line'].unique())
            for line, grp in combos.groupby('line', sort=False):
                for i, pert in enumerate(grp['pert'].tolist()):
                    self._combo_row_of[(line, pert)] = i

    def __len__(self):

        return len(self.perturbations) * 1000

    def __getitem__(self, idx):
        # 随机选一个 perturbation
        perturbation_idx = np.random.choice(len(self.perturbations), 1)[0]
        perturbation_id = self.perturbations[perturbation_idx]

        if self.sampler._line_aware:
            # same-line pairing (see TrainSampler.get_batch)
            elig = self.sampler._eligible_lines[perturbation_id]
            line = elig[np.random.randint(len(elig))]
            tgt_idx = self.sampler._tgt_pool[(perturbation_id, line)]
            src_idx = self.sampler._ctl_pool[line]
        else:
            line = None
            tgt_idx = (self.sampler.adata.obs['perturbation_covariates'] == perturbation_id).to_numpy().nonzero()[0]
            src_idx = self.control_idx
        tgt_batch_idx = np.random.choice(tgt_idx, self.batch_size)
        src_batch_idx = np.random.choice(src_idx, self.batch_size)
        src_batch = torch.from_numpy(_row_slice_csr(self.sampler.adata.X, src_batch_idx).toarray())
        tgt_batch = torch.from_numpy(_row_slice_csr(self.sampler.adata.X, tgt_batch_idx).toarray())

        # 残差目标查表键（pert 条件串 'GENE+control' -> 'GENE'）
        line_id = self._lines_list.index(line) if line is not None else -1
        combo_row = self._combo_row_of.get((line, perturbation_id.split('+')[0]), -1)

        return {
            'src_cell_data': src_batch,
            'tgt_cell_data': tgt_batch,
            'src_cell_id': list(self.sampler.cells_name[src_batch_idx]),
            'tgt_cell_id': list(self.sampler.cells_name[tgt_batch_idx]),
            'condition_id': torch.tensor(self.sampler.perturbation_covariates_id[tgt_batch_idx]),
            'line_id': torch.tensor([line_id], dtype=torch.long),
            'combo_row': torch.tensor([combo_row], dtype=torch.long),
        }
class BinDiscretizer:
    """
    data = np.random.exponential(scale=2.0, size=1000)

    bd = BinDiscretizer(n_bins=200)
    bd.fit(data)

    bd.save_edges('./data/combosciplex/bin_discretizer_edges.pkl')

    new_bd = BinDiscretizer(n_bins=200)
    new_bd.load_edges('./data/combosciplex/bin_discretizer_edges.pkl')

    binned = bd.transform(data)

    recon = bd.inverse_transform(binned, random=False)
    
    Note: 0 is treated as a separate class (class 0), and non-zero values are discretized into classes 1 to n_bins.
    """
    def __init__(self, n_bins: int, strategy: str = "quantile"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges = None  # will be (n_bins + 1, ) array

    def fit(self, data: Union[np.ndarray, torch.Tensor]):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.flatten()
        data = data[data > 0]  # exclude zeros from fitting

        if len(data) == 0:
            raise ValueError("No non-zero entries in data to fit.")

        if self.strategy == "quantile":
            self.edges = np.quantile(data, np.linspace(0, 1, self.n_bins + 1))
        elif self.strategy == "uniform":
            self.edges = np.linspace(data.min(), data.max(), self.n_bins + 1)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.edges is None:
            raise RuntimeError("Call fit() before transform().")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            orig_dtype = data.dtype
            data = data.detach().cpu().numpy()

        out = np.zeros_like(data, dtype=np.int64)
        mask = data > 0

        # Digitize non-zero entries: 0 remains 0, non-zero values get classes 1 to n_bins
        if np.any(mask):
            # np.digitize returns 0-based indices for the bins
            # We want to map these to 1-based class indices
            digitized = np.digitize(data[mask], self.edges[1:-1])
            # Convert 0-based bin indices to 1-based class indices
            # digitized=0 means it's in the first bin, which should be class 1
            # digitized=1 means it's in the second bin, which should be class 2, etc.
            out[mask] = digitized + 1

        if is_torch:
            return torch.from_numpy(out).to(dtype=torch.int64)
        return out

    def inverse_transform(self, digitized: Union[np.ndarray, torch.Tensor], random: bool = False) -> Union[np.ndarray, torch.Tensor]:
        if self.edges is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        is_torch = isinstance(digitized, torch.Tensor)
        if is_torch:
            orig_dtype = digitized.dtype
            digitized = digitized.detach().cpu().numpy()

        out = np.zeros_like(digitized, dtype=np.float64)
        mask = digitized > 0

        if np.any(mask):
            ids = digitized[mask]
            # Ensure ids are within valid range (1 to n_bins)
            ids = np.clip(ids, 1, self.n_bins)
            # Convert 1-based class indices back to 0-based bin indices
            bin_ids = ids - 1
            lefts = self.edges[bin_ids]
            rights = self.edges[bin_ids + 1]

            if random:
                out[mask] = np.random.uniform(lefts, rights)
            else:
                out[mask] = (lefts + rights) / 2

        if is_torch:
            return torch.from_numpy(out).to(dtype=torch.float64)
        return out
    
    def save_edges(self, filepath: Union[str, Path]):
        """Save edges to a file"""
        if self.edges is None:
            raise RuntimeError("No edges to save. Call fit() first.")
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump({'edges': self.edges, 'n_bins': self.n_bins}, f)
            
    def load_edges(self, filepath: Union[str, Path]):
        """Load edges from a file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            loaded_n_bins = data['n_bins']
            if loaded_n_bins != self.n_bins:
                raise ValueError(f"Loaded n_bins ({loaded_n_bins}) does not match initialized n_bins ({self.n_bins})")
            self.edges = data['edges']
    
class PretrainData(Dataset):
    def __init__(self, adata: sc.AnnData, drug_dict: dict):
        self.adata = adata
        self.drug_dict = drug_dict
        self.X = torch.from_numpy(adata.X.toarray())
        self.cell_id = adata.obs_names
        # self.drug1 = torch.tensor(np.array(adata.obs['Drug1'].apply(lambda x: drug_dict[x])))
        # self.drug2 = torch.tensor(np.array(adata.obs['Drug2'].apply(lambda x: drug_dict[x])))
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        return {
            'values' : self.X[idx], 
            'cell_id': self.cell_id[idx],
        }
    
            
if __name__ == "__main__":
    data = Data(data_path='./data')
    data.load_data(data_name='combosciplex')
    data.process_data()
    
    
    