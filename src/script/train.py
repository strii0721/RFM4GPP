import accelerate
import torch
import torch.nn as nn
import tyro
import glob
import shutil
import math
from src.utils.config_utils import ConfigUtils, FlowConfig as Config
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import random
from src.data_process.data import Data, PerturbationDataset
from src.script.build_cache import build_cache
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import ODESolver
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.models.perturbation.moduls import PerturbationEmbedding
import pdb
from src.flow_matching.path.scheduler import CondOTScheduler
import scanpy as sc
import os
from src.data_process.utils import build_generated_anndata

import json
from accelerate import Accelerator,DistributedDataParallelKwargs
import datetime
import torchdiffeq
from tqdm import trange
import numpy as np
import anndata as ad
import pandas as pd
import sys
from src.utils.utils import save_checkpoint, load_checkpoint, make_lognorm_poisson_noise, pick_eval_score, process_vocab, set_requires_grad_for_p_only, get_perturbation_emb

ot_sampler = OTPlanSampler(method="exact") 
path = AffineProbPath(scheduler=CondOTScheduler())

# 训练每步基因选择状态（main() 初始化，train_step 读取）：
# 2026-09-21 用户定案——建模基因子集 = 完整基因轴（固定集合，含 300 panel 列；
# 推理侧仅对扰动自身靶列置 0），每步从池随机抽 L=infer_top_gene 个基因。
_pool_idx: torch.Tensor | None = None   # 随机抽样池（缓存列位置）
# AE 可学习编解码器（2026-10-06 warm-start + 联合微调；train_step 引用）
_ae_mode = False
_ae_enc = None
_ae_dec = None
_latent_ids: torch.Tensor | None = None

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma**2)
    dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-beta * dist)

def mmd_loss(pred, tgt, sigma=1.0):
    xx = gaussian_kernel(pred, pred, sigma).mean(dim=(1))
    yy = gaussian_kernel(tgt, tgt, sigma).mean(dim=(1))
    xy = gaussian_kernel(pred, tgt, sigma).mean(dim=(1))
    return (xx + yy - 2 * xy).mean()

def pairwise_sq_dists(X, Y):
    # X:[m,d], Y:[n,d] -> [m,n]
    return torch.cdist(X, Y, p=2)**2

@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    Z = X
    D2 = pairwise_sq_dists(Z, Z)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)          
    s2 = torch.tensor(scales, device=Z.device) * m 
    sigmas = torch.sqrt(s2)                
    return [float(s.item()) for s in sigmas]

def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    """
    """
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)   # [m,m]
    Dyy = pairwise_sq_dists(Y, Y)   # [n,n]
    Dxy = pairwise_sq_dists(X, Y)   # [m,n]

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()  # / (m*n)
        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return torch.stack(vals).mean()

def train_step(source, target, main_effect_row, perturbation_id, vf, criterion, accelerator, noise_type='Poisson', mode="predict_y"):
    B = source.shape[0]
    device = accelerator.device
    
    # 2026-09-21 用户定案：建模基因子集 = 完整基因轴（含 panel），每步从池随机抽
    # L=infer_top_gene 个基因（池大小 11,371 时 = 全轴）
    assert _pool_idx is not None, \
        'sampling pool not initialized (main() sets it before training)'
    n_rand = min(config.infer_top_gene, _pool_idx.shape[0])
    rand = torch.randperm(_pool_idx.shape[0], device=device)[:n_rand]
    input_gene_ids = _pool_idx[rand]
    source = source[:,input_gene_ids]
    target = target[:,input_gene_ids]
    gene = gene_ids.repeat(B,1).to(device)
    gene_input = gene[:,input_gene_ids]
    
    if mode=="predict_y" and _ae_mode:
        # AE 方案（2026-10-06 warm-start + 联合微调）：CFM 流匹配搬到潜空间，
        # 监督目标只在基因空间（潜端点经 dec 回投影 vs Res——梯度穿 dec→主模型→enc）。
        t = torch.rand(B, device=device)
        # 细胞对级 Res（2026-10-08 定案）：数据层 OT 配对对照/扰动细胞 →
        # r_pair = log2 fold（log1p 空间差 ÷ ln2）→ 减去 pseudobulk 主效应
        src_p, tgt_p = ot_sampler.sample_plan(source, target)
        res_cell = (tgt_p - src_p) / math.log(2.0) - main_effect_row[None, :]
        z_s = _ae_enc(src_p)                                     # 源潜 (B, latent)
        z1 = _ae_enc(res_cell)                                   # 目标潜端点（梯度回传）
        z0 = torch.randn_like(z1)
        z0, z1 = ot_sampler.sample_plan(z0, z1)                  # 路径层 OT（潜空间）
        z_t = (1 - t)[:, None] * z0 + t[:, None] * z1           # 直线插值
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=getattr(config, 'use_bf16', False)):
            lid = _latent_ids.unsqueeze(0).expand(B, -1)   # (B, latent)——GeneEncoder 需 batch 维
            v_pred = vf(lid, z_t, t, z_s, perturbation_id, lid, mode=mode)
        loss_cfm = ((v_pred - (z1 - z0)) ** 2).mean()           # 潜空间 CFM 速度匹配
        z1_hat = z_t + v_pred * (1 - t)[:, None]                # 端点估计（潜）
        res_hat = _ae_dec(z1_hat)                                # (B, n_genes)
        loss = loss_cfm + F.mse_loss(res_hat, res_cell)         # 基因空间端点（穿 dec）
        if config.use_mmd_loss:
            # 细胞对级 MMD（2026-10-08 加回）：Res_cell 是真实细胞残差分布，多尺度 RBF
            # 匹配分布形状（补 MSE 均值回归盲区）；梯度穿 dec→主模型→enc（联合微调）
            sigmas = median_sigmas(res_cell.float(), scales=(0.5, 1.0, 2.0, 4.0))
            loss = loss + mmd2_unbiased_multi_sigma(res_hat.float(), res_cell.float(), sigmas) * config.gamma
        return loss

    if mode=="predict_y":
        t = torch.rand(B, device=device)
        # 细胞对级 Res（2026-10-08 用户定案，双层 OT）：
        # ① 数据层 OT：配对 source↔target，r_pair = log2 fold（log1p 空间差 ÷ ln2）
        # ② 监督目标 x₁ = Res_cell = r_pair − (r̄_c + r̄_p − ḡ)（pseudobulk 主效应剥离）
        # ③ 路径层 OT：噪声 x₀ 与 Res_cell 端点再配对（上游 OT-CFM 语义，路径短直）
        src_p, tgt_p = ot_sampler.sample_plan(source, target)
        res_cell = (tgt_p - src_p) / math.log(2.0) - main_effect_row[None, :]
        source_cond = src_p[:, input_gene_ids]
        res_b = res_cell[:, input_gene_ids]
        target_noise = torch.randn_like(res_b)
        x0, x1 = ot_sampler.sample_plan(target_noise, res_b)
        path_x1 = path.sample(t=t, x_0=x0, x_1=x1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=getattr(config, 'use_bf16', False)):
            predicted_x_t_velocity = vf(gene_input,path_x1.x_t, path_x1.t,source_cond,perturbation_id, gene_input, mode=mode)
        loss = ((predicted_x_t_velocity - path_x1.dx_t)**2).mean()
        
        if config.use_mmd_loss:
            x1_hat = path_x1.x_t + predicted_x_t_velocity*(1-t).unsqueeze(-1)
            # fp32 for stable pairwise-distance kernels under bf16 training
            x1_hat_f = x1_hat.float()
            target_f = x1.float()
            sigmas = median_sigmas(target_f, scales=(0.5,1.0,2.0,4.0))
            _mmd_loss = mmd2_unbiased_multi_sigma(x1_hat_f, target_f, sigmas)
            loss = loss + _mmd_loss * config.gamma

    elif mode=="predict_p":
        t_p = torch.ones(B, device=device)  # Or uniform(0.7,1.0)
        predicted_p_embed = vf(gene_input, target, t_p, source, perturbation_id, gene_input, mode=mode)
        if hasattr(vf, "module"):
            base_vf = vf.module
        else:
            base_vf = vf
        p_embed_gt = base_vf.get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt  = F.normalize(p_embed_gt.detach(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()  # cosine distance
    
    return loss

@torch.inference_mode()
def test(data_sampler, vf, accelerator,  batch_size=128, path='./',vocab=None,scheme='mse', max_pairs=None):
    gene_ids_test = vocab.encode(list(data_sampler.adata.var_names))
    
    gene_ids_test = torch.tensor(gene_ids_test, dtype=torch.long, device=device)
    # line-consistent eval: each (perturbation, cell line) pair is scored against
    # its OWN line's control/target pools (see TestDataset.perturbation_line_pairs)
    pairs = data_sampler.perturbation_line_pairs(min_tgt_cells=1)
    if max_pairs:
        pairs = pairs[:max_pairs]
    elif config.max_test_perts:
        pairs = pairs[:config.max_test_perts]
    line_aware = data_sampler._line_aware
    lines = sorted({L for _, L in pairs}) if line_aware else [None]
    print(f'eval pairs: {len(pairs)} across {len(lines)} lines')
    per_line_scores = []
    for L in lines:
        pair_L = [(p, l) for p, l in pairs if l == L]
        control_data = data_sampler.get_control_data(line=L)
        all_pred, all_tgt = [control_data['src_cell_data']], [control_data['src_cell_data']]
        obs_p = ['control'] * control_data['src_cell_data'].shape[0]
        obs_r = ['control'] * control_data['src_cell_data'].shape[0]
        for perturbation_name, _ in pair_L:
            perturbation_data = data_sampler.get_perturbation_data(perturbation_name, line=L)
            target = perturbation_data['tgt_cell_data']
            perturbation_id = perturbation_data['condition_id']
            source = control_data['src_cell_data'].to(device)
            perturbation_id = perturbation_id.to(device)
            if config.perturbation_function == 'crisper':
                # 单槽扰动条件（方案B 2026-09-14）：与训练 loop 一致，只编码目标基因
                perturbation_name_crisper = [inverse_dict[int(perturbation_id[0, 0].cpu().item())]]
                perturbation_id = torch.tensor(vocab.encode(perturbation_name_crisper), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source.shape[0], 1)
            
            idx = torch.randperm(source.shape[0])
            source = source[idx]
            N = 128
            source = source[:N]
            perturbation_id = perturbation_id[:N]
            
            pred_expressions = []
            for i in trange(0, N, batch_size):
                batch_perturbation_id = perturbation_id[i:i+batch_size]
                
                batch_perturbation_id = batch_perturbation_id.to(accelerator.device)
                
                pred_expression = generate_sample(wrapped_vf,source[i:i+batch_size],batch_perturbation_id,vf,gene_ids=gene_ids_test,gene_all=gene_ids_test)
                pred_expressions.append(pred_expression)
                
            pred_expressions = torch.cat(pred_expressions, dim=0).cpu().numpy()
            all_pred.append(pred_expressions)
            all_tgt.append(target)
            obs_p.extend([perturbation_name] * pred_expressions.shape[0])
            obs_r.extend([perturbation_name] * target.shape[0])

        all_pred_expressions = np.concatenate(all_pred, axis=0)
        all_target_expressions = np.concatenate(all_tgt, axis=0)
        obs_pred = pd.DataFrame({'perturbation':obs_p})
        obs_real = pd.DataFrame({'perturbation':obs_r})
        pred = ad.AnnData(X=all_pred_expressions, obs=obs_pred)
        real = ad.AnnData(X=all_target_expressions, obs=obs_real)
        
        if accelerator.is_main_process:
            line_tag = L if L is not None else 'all'
            pred.write_h5ad(os.path.join(path, f'pred_{line_tag}.h5ad'))
            real.write_h5ad(os.path.join(path, f'real_{line_tag}.h5ad'))
            # 官方 vcc2026 六指标（cell-eval2 0.16 CLI，取代旧 cell_eval 2025 指标集；
            # 旧包与 pdex>=0.3 不兼容已于 2026-09-14 弃用）。lognorm 输入 = log1p(CP10k)。
            import subprocess
            ce2_bin = os.path.join(os.path.dirname(sys.executable), 'cell-eval2')
            rdir = os.path.join(path, f'ce2_{line_tag}')
            subprocess.run(
                [ce2_bin, 'run',
                 '-ap', os.path.join(path, f'pred_{line_tag}.h5ad'),
                 '-ar', os.path.join(path, f'real_{line_tag}.h5ad'),
                 '--preset', 'vcc2026', '--input-type', 'lognorm',
                 '--pert-col', 'perturbation', '--control', 'control',
                 '--set', 'de.backend=pdex', '-o', rdir],
                check=True,
            )
            agg = pd.read_csv(os.path.join(rdir, 'agg_results.csv'))
            per_line_scores.append((line_tag, pick_eval_score(agg, scheme)))

    eval_score = None
    if accelerator.is_main_process and per_line_scores:
        eval_score = float(np.mean([s for _, s in per_line_scores]))
        for tag, s in per_line_scores:
            print(f'eval[{tag}]: {scheme} = {s:.4f}')
        print(f'eval mean over lines: {eval_score:.4f}')
    
    return eval_score

def wrapped_vf(target,t,source,perturbation_id,vf,gene_ids, gene_all):
    
    gene = gene_ids.repeat(source.shape[0],1).to(device)
    predicted_x_t_velocity = vf(gene,target,t,source,perturbation_id,gene_all)
    
    return predicted_x_t_velocity

@torch.no_grad()
def generate_sample(wrapped_vf,source,condition_vec=None,vf=None,gene_ids=None,gene_all=None,steps=100,method="euler"):
    
    noise_type = config.noise_type
    if noise_type=="Gaussian":
        # 噪声维=实际窗口维（source.shape[1]），勿用 config.infer_top_gene——
        # 全轴定案后 config 值(11919)≠运行时窗口(非 panel 缓存列 11,071)
        target_noise = torch.randn(source.shape[0], source.shape[1], device=source.device)
    elif noise_type=="Poisson":
        target_noise = make_lognorm_poisson_noise(
            target_log=source,
            alpha=getattr(config, "poisson_alpha", 0.8),           
            per_cell_L=getattr(config, "poisson_target_sum", 1e4), 
        )
        
    traj = torchdiffeq.odeint(lambda t,x: wrapped_vf(x,t,source,condition_vec,vf,gene_ids,gene_all),
                              target_noise,
                              torch.linspace(0,1,steps).to(source.device),
                              atol=1e-4,
                              rtol=1e-4,
                              method=method)
    # t = torch.linspace(0,1,steps).to(source.device)
    # traj = [target_noise + 0.8*wrapped_vf(target_noise,t,source,condition_vec,vf,gene_ids,gene_all)]
    
    return torch.clamp(traj[-1], min=0)
    
if __name__ == "__main__":
    # ---- 训练启动入口（2026-09-28 并入原 scripts/train.sh 职责）----
    # 环境准备：expandable_segments（全轴 B=2 反向峰值 73GB+7.7GB 瞬时分配，
    # 不设置时碎片浪费 ~9.8GB 越 80GB 墙 OOM，torch 2.7 支持）+ 行缓冲 stdout
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    # 时间戳 = 实验名（output/train_<ts>/ 与 logs/train_<ts>/ 同名对应；
    # rank0 建日志目录并把 stdout/stderr 落 train.log）
    ts = os.environ.get('SCDFM_RUN_TS') or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    os.environ['SCDFM_RUN_TS'] = ts
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    config = ConfigUtils.load(Config)
    # 2026-10-07：batch_size_per_gpu 即每 rank batch，不再由 batch_total 分摊

    if rank == 0:
        log_dir = os.path.join(config.log_base_dir, f'train_{ts}')
        os.makedirs(log_dir, exist_ok=True)
        log_f = open(os.path.join(log_dir, 'train.log'), 'a', buffering=1)
        os.dup2(log_f.fileno(), 1)
        os.dup2(log_f.fileno(), 2)
        sys.stdout = log_f
        sys.stderr = log_f
        # 缓存/共表达 mask/vocab 必须先单进程预构建（逻辑在 src/script/build_cache.py，
        # 2026-09-28 拆出；8 个 DDP rank 并发写同一 h5ad 会在 NFS 上撞 h5py 文件锁，
        # 实测 BlockingIOError errno 11）。rank0 构建期间其余 rank 阻塞在下方
        # init_process_group，天然成为 barrier。
        build_cache(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs]
    )
    # 2026-09-17：8 rank 启动路径（87GB 缓存读入 + TrainSampler 建池）耗时差可达
    # 10+ 分钟，prepare() 时才 init process group 会触发 NCCL 600s 超时（实测
    # rank2 等 rank0 的 ncclUniqueId 超时）。重活开始前先 rendezvous（各 rank
    # spawn 后数秒内齐达）；accelerate prepare() 检测已初始化会跳过（state.py
    # is_initialized 守卫）。timeout 放大到 2h 防后续 DDP 广播因最慢 rank 滞后超时。
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://',
                timeout=datetime.timedelta(hours=2),
                device_id=torch.device(f'cuda:{local_rank}'))
        # 关键 1/2：首个 collective 强制 NCCL communicator 惰性创建完成（2026-09-17
        # 两次实测：先到的 rank 在 prepare/DDP 的首个 collective 处向 store 取
        # rank0 的 ncclUniqueId，600s 超时——rank0 还在读缓存没到 collective）。
        # 在重活开始前 barrier，所有 rank 数秒内齐达，comm 一次建好。
        torch.distributed.barrier()
    if accelerator.is_main_process:
        print(config)
        save_path = config.make_path()
        os.makedirs(save_path, exist_ok=True)
    device = accelerator.device
    
    data_manager = Data(config.data_path, config=config)

    data_manager.load_data(config.data_name)
    data_manager.process_data(n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene, split_method=config.split_method, fold=config.fold, use_negative_edge=config.use_negative_edge, k=config.topk)
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size_per_gpu)
    
    train_dataset = PerturbationDataset(train_sampler, config.batch_size_per_gpu,
                                        residual_dir=config.frozen_tensors_dir)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=config.num_workers,pin_memory=True,persistent_workers=True)  # batch_size=1 因为每个getitem本身就是一个batch
    # data.py computes the (per-corpus / per-fold) mask path and exposes it;
    # recomputing here would drift from the file actually built in process_data
    if hasattr(data_manager, 'mask_path'):
        mask_path = data_manager.mask_path
    elif config.use_negative_edge:
        mask_path = os.path.join(data_manager.data_path, data_manager.data_name, 'mask_fold_' + str(config.fold) + 'topk_' + str(config.topk) + config.split_method + '_negative_edge' + '.pt')
    else:
        mask_path = os.path.join(data_manager.data_path, data_manager.data_name, 'mask_fold_' + str(config.fold) + 'topk_' + str(config.topk) + config.split_method + '.pt')
    vf = instantiate_model(config.model_type,
                           ntoken = config.resolve_ntoken(),
                           d_model = config.d_model,
                           d_perturbation = config.d_model,
                           fusion_method = config.fusion_method,
                           perturbation_function = config.perturbation_function,
                           mask_path = mask_path,
                           pert_ntoken = config.resolve_pert_ntoken(),
                           # AE 模式：潜 token 与基因共表达 mask 尺寸不符 → 关
                           use_perturbation_interaction = config.use_perturbation_interaction and not config.ae_encoder_ckpt,
                           )
    
    model_path = config.make_path()

    # AE warm-start（2026-10-06 用户定案）：加载可学习编解码器、不冻结（联合微调）。
    _ae_mode = bool(config.ae_encoder_ckpt and config.ae_decoder_ckpt)
    if _ae_mode:
        from src.models.autoencoder import AEEncoder, AEDecoder
        _ae_enc = AEEncoder(config.n_top_genes, 8192, config.ae_latent_dim).to(device)
        _ae_dec = AEDecoder(config.n_top_genes, 8192, config.ae_latent_dim).to(device)
        _ae_enc.load_state_dict(torch.load(config.ae_encoder_ckpt, map_location='cpu'))
        _ae_dec.load_state_dict(torch.load(config.ae_decoder_ckpt, map_location='cpu'))
        _latent_ids = torch.arange(config.ae_latent_dim, dtype=torch.long, device=device)
        print(f'[AE] warm-start loaded: {config.ae_encoder_ckpt} / '
              f'{config.ae_decoder_ckpt} latent={config.ae_latent_dim}（不冻结，联合微调）',
              flush=True)

    vocab = process_vocab(data_manager, config)

    gene_ids = vocab.encode(list(data_manager.adata.var_names))
    
    gene_ids = torch.tensor(gene_ids, dtype=torch.long, device=device)

    # 残差目标表（范式二，2026-09-26）：(line, pert) 冻结产物，列对齐缓存基因轴；
    # mmap 懒加载（res_K562.npy 395MB 不常驻），每 rank 各持一份共享页。
    # 2026-10-08 用户定案（细胞对级 Res + 双层 OT）：res_<line>.npy 组合级表闲置，
    # 监督改为 OT 配对的细胞对残差 = r_pair − (r̄_c+r̄_p−ḡ)；三冻结向量 pseudobulk 口径不变。
    _res_lines = sorted(pd.read_csv(os.path.join(config.frozen_tensors_dir, 'combos.csv'))['line'].unique())
    _rbar_c_all = np.load(os.path.join(config.frozen_tensors_dir, 'rbar_c.npy'),
                          mmap_mode='r')   # (n_lines, 19657)
    _rbar_c_lines = pd.read_csv(os.path.join(config.frozen_tensors_dir, 'rbar_c_lines.csv'))['line'].tolist()
    _rbar_c_rows = {L: np.array(_rbar_c_all[_rbar_c_lines.index(L)], dtype=np.float32)
                    for L in _rbar_c_lines}
    _rbar_p_all = np.load(os.path.join(config.frozen_tensors_dir, 'rbar_p.npy'),
                          mmap_mode='r')   # (n_perts, 19657)
    _rbar_p_perts = pd.read_csv(os.path.join(config.frozen_tensors_dir, 'rbar_p_perts.csv'))['pert'].tolist()
    _rbar_p_rows = {p: np.array(_rbar_p_all[i], dtype=np.float32)
                    for i, p in enumerate(_rbar_p_perts)}
    _gbar = np.load(os.path.join(config.frozen_tensors_dir, 'gbar.npy')).astype(np.float32)
    assert _gbar.shape[0] == gene_ids.shape[0], 'gbar != cache gene axis'
    print(f'##### residual targets loaded: lines={_res_lines}, '
          f'rbar_c={len(_rbar_c_lines)} rbar_p={len(_rbar_p_perts)} #####', flush=True)

    # 训练每步基因选择的采样池（2026-09-21 用户定案）：建模基因子集 = 完整基因轴
    # （固定集合，含 panel；panel 基因是其他扰动的真实 DEG，不可排除）。
    # 默认池 = train_pool_path 的基因清单；空串回退 = 全部缓存列（11,371，含 300 panel）。
    panel_raw = pd.read_csv(config.panel_csv_path, header=None)[0].astype(str).tolist()
    panel_genes = [g for g in panel_raw if g in set(data_manager.adata.var_names)]
    panel_ids = set(vocab.encode(panel_genes))
    panel_mask = torch.tensor([int(g) in panel_ids for g in gene_ids.tolist()],
                              dtype=torch.bool, device=device)
    if config.train_pool_path:
        pool_raw = pd.read_csv(config.train_pool_path)['gene_name'].astype(str).tolist()
        var_names = list(data_manager.adata.var_names)
        pool_genes = [g for g in pool_raw if g in set(var_names)]
        assert pool_genes, f'train_pool_path={config.train_pool_path!r} yields no usable genes'
        pos_map = {g: i for i, g in enumerate(var_names)}
        _pool_idx = torch.tensor([pos_map[g] for g in pool_genes],
                                 dtype=torch.long, device=device)
        print(f'##### training sampling pool: {_pool_idx.shape[0]} genes from '
              f'{config.train_pool_path} (panel included) #####', flush=True)
    else:
        _pool_idx = torch.arange(gene_ids.shape[0], device=device)
        print(f'##### full-axis training window: per-step L={config.infer_top_gene} from '
              f'{_pool_idx.shape[0]} cache cols ({int(panel_mask.sum())} panel cols '
              f'included) #####', flush=True)
    
    save_path = config.make_path()
    best_loss = float('inf')
    
    criterion = nn.MSELoss()
    # AE 联合微调：encoder/decoder 参数并入 optimizer（不冻结；dec 梯度强、enc 隔主模型弱）
    _ae_params = (list(_ae_enc.parameters()) + list(_ae_dec.parameters())) if _ae_mode else []
    optimizer = torch.optim.Adam(list(vf.parameters()) + _ae_params, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.eta_min)
    
    ckpt_path = config.resume
    if ckpt_path:
        # 续训（2026-10-05 用户定案 --resume <ckpt>）：加载 model+optimizer；
        # 调度器重建——旧调度 last_epoch 已退至 eta_min，直接继承会全程 lr≈1e-6
        # 训不动；重建后 lr 从 base 按 config.steps 重新退火，last_epoch=start_iteration。
        start_iteration, _ = load_checkpoint(ckpt_path, vf, optimizer, scheduler=None,
                                             ae_enc=_ae_enc if _ae_mode else None,
                                             ae_dec=_ae_dec if _ae_mode else None)
        start_iteration += 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.steps, eta_min=config.eta_min)
        scheduler.last_epoch = start_iteration
        print(f'[resume] {ckpt_path} -> start_iteration={start_iteration}, '
              f'scheduler rebuilt (T_max={config.steps})', flush=True)
    else:
        start_iteration = 0
    # 关键 2/2（2026-09-17 实测）：DDP 构造的 _verify_params_across_processes 用
    # store 交换各 rank 参数数——rank 间加载差可达 37+ 分钟，先到的 rank 读到未
    # 就绪 rank 的空槽（"Rank 2 has inconsistent 0 params"）直接报错。prepare 前
    # 第二次 rendezvous：等最慢 rank 一起进 DDP。
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        torch.distributed.barrier()
    vf = accelerator.prepare(vf)
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer,scheduler,dataloader)
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}
    iteration = start_iteration
    # 早停（2026-10-05 用户定案：steps 10k→100k）：以 print_every 窗口平均 loss 为信号
    # （DDP 下 do_eval 必 False——NCCL 死锁，不能用验证集指标），patience 个窗口无改善
    # 提前停；best 窗口另存 best_checkpoint.pt。patience=0 禁用。
    window_loss, window_cnt = 0.0, 0
    best_window = float('inf')
    bad_cnt = 0
    early_stop = False
    while iteration < config.steps and not early_stop:
        for batch_data in dataloader:
            
            source = batch_data['src_cell_data'].squeeze(0)
            it_t0 = time.time()   # 本 iteration 计时（打印处算耗时，2026-09-26 用户定案）
            target = batch_data['tgt_cell_data'].squeeze(0)
            perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
            if config.perturbation_function == 'crisper':
                # 单槽扰动条件（2026-10-06）：VCC 单基因 CRISPRi，只编码 target_gene
                # （上游双药物槽位 Drug1/Drug2 已删）
                # 提交侧 generate_submission.py 本就是单槽 (B,1)，改后两侧对齐。
                perturbation_name = [inverse_dict[int(perturbation_id[0, 0].cpu().item())]]
                perturbation_id = torch.tensor(vocab.encode(perturbation_name), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source.shape[0], 1)

            # 主效应行（2026-10-08 定案）：r̄_c(line) + r̄_p(pert) − ḡ（pseudobulk 加性主效应，
            # 全轴 19657）；细胞对残差在 train_step 内 = r_pair(OT 配对) − 主效应
            line_id = int(batch_data['line_id'].squeeze(0).item())
            _line = _res_lines[line_id]
            _pert_name = perturbation_name[0]
            _main_effect = (_rbar_c_rows.get(_line, np.zeros(_gbar.shape[0], np.float32))
                            + _rbar_p_rows.get(_pert_name, np.zeros(_gbar.shape[0], np.float32))
                            - _gbar)
            main_effect = torch.from_numpy(_main_effect).to(device)

            set_requires_grad_for_p_only(vf, p_only=config.mode)
            loss = train_step(source, target, main_effect, perturbation_id, vf, criterion, accelerator, noise_type=config.noise_type, mode=config.mode)
            if accelerator.is_main_process:
                window_loss += loss.item()
                window_cnt += 1
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            
            if iteration % config.print_every == 0:
                # lockstep: all ranks wait here so main-process checkpoint save
                # does not let other ranks race ahead and exit
                accelerator.wait_for_everyone()
                # 早停窗口判定（主进程；iteration=0 窗口仅 1 步，跳过不清零）
                improved = False
                if accelerator.is_main_process and window_cnt > 1:
                    avg_w = window_loss / window_cnt
                    improved = avg_w < best_window - 1e-6
                    if improved:
                        best_window = avg_w
                        bad_cnt = 0
                    elif config.early_stop_patience > 0:
                        bad_cnt += 1
                    print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] [window] '
                          f'avg_loss={avg_w:.5f} best={best_window:.5f} '
                          f'bad={bad_cnt}/{config.early_stop_patience}', flush=True)
                    window_loss, window_cnt = 0.0, 0
                save_path_ = os.path.join(save_path, f'iteration_{iteration}')
                os.makedirs(save_path_, exist_ok=True)
                eval_score = None
                if accelerator.is_main_process:
                    print(f"svaing {iteration}'s checkpoint...")
                    save_checkpoint(
                        model=accelerator.unwrap_model(vf), 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        iteration=iteration, 
                        eval_score=None,  # 不需要评估分数
                        save_path=save_path_, 
                        is_best=improved,
                        ae_enc=_ae_enc if _ae_mode else None,
                        ae_dec=_ae_dec if _ae_mode else None,
                        best_dir=save_path,
                    )
                    # 检查点保留策略（2026-10-06 用户定案）：仅保留近 5 个 + 最好的一个。
                    # best 三文件由 save_checkpoint 写 best/ 子目录（不在 iteration_N 内混副本）。
                    _its = sorted(glob.glob(os.path.join(save_path, 'iteration_*')),
                                  key=lambda p: int(os.path.basename(p).split('_')[1]))
                    for _old in _its[:-5]:
                        shutil.rmtree(_old, ignore_errors=True)
                # 早停触发广播：主进程判定 → all_reduce 全体统一退出（防 DDP 死锁）
                stop_sig = torch.tensor([1.0 if (accelerator.is_main_process and
                                                 config.early_stop_patience > 0 and
                                                 bad_cnt >= config.early_stop_patience)
                                        else 0.0], device=device)
                if int(os.environ.get('WORLD_SIZE', '1')) > 1:
                    torch.distributed.all_reduce(stop_sig)
                early_stop = stop_sig.item() >= 1.0
                if accelerator.is_main_process and early_stop:
                    print(f'[early stop] {config.early_stop_patience} windows without '
                          f'improvement -> stop at {iteration} (best={best_window:.5f})',
                          flush=True)
                if config.do_eval:
                    # NOTE: in-loop eval deadlocks under multi-GPU DDP (other ranks'
                    # first backward all_reduce waits for the main rank while it
                    # evaluates). Only safe for single-GPU runs; DDP training must
                    # use --no-do_eval and evaluate from checkpoints afterwards.
                    if accelerator.is_main_process:
                        eval_score = test(valid_sampler, vf, accelerator, batch_size=config.batch_size_per_gpu, path=save_path_,vocab=vocab)
                
            accelerator.wait_for_everyone()
            
            # 2026-09-26 用户定案：每 iteration 只打一行（主进程），
            # 废弃 tqdm 进度条（8 rank 各写一行 + \r 刷屏）；
            # 格式：loss + 进度 n/N + 本 iteration 耗时（含 checkpoint 保存等事件）
            if accelerator.is_main_process:
                print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] '
                      f'loss: {loss.item():.4f}, iteration: {iteration}/{config.steps}, '
                      f'{time.time() - it_t0:.2f}s/it', flush=True)
            iteration += 1
            if iteration >= config.steps or early_stop:
                break
            
    # save final checkpoint if the loop ended without saving this iteration
    # (covers both between-marks endings and exact-multiple endings like
    #  steps=5000, print_every=1000 -> iteration_5000)
    final_dir = os.path.join(save_path, f'iteration_{iteration}')
    if not os.path.exists(os.path.join(final_dir, 'checkpoint.pt')):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(final_dir, exist_ok=True)
            print(f"svaing final {iteration}'s checkpoint...")
            save_checkpoint(
                model=accelerator.unwrap_model(vf),
                optimizer=optimizer,
                scheduler=scheduler,
                iteration=iteration,
                eval_score=None,
                save_path=final_dir,
                is_best=False,
            )
            