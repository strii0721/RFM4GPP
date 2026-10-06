#!/usr/bin/env python3
"""VCC-2026 submission generator（log1p 单空间，2026-09-11 重构）。

加载训练 ckpt，从官方 A/B/C 对照流式生成 300×3×(400 细胞) 的提交物，
分片并行（SHARD_ID / NUM_SHARDS env），由 scripts/remote_benchmark.sh 之前的
本地推理/编排流程使用（2026-09-28 起为辅助脚本，与 tmp/ 同仓随代码同步）。

每 (context, perturbation)：
  source = 400 个对照细胞（稳定种子抽样）
  pred_log1p = ODE(x0~noise, cond = source 建模基因 + 扰动基因)
  full_log1p = 建模基因 <- pred、其余基因 <- 对照 log1p(CP10k)
  lam = expm1(clip(full_log1p)) → 按源细胞深度缩放 → Poisson 出整数计数

缓存/词表/共表达图一律【只读】cache/vcc/ 与 src/tokenizer/（与训练同名派生，
缺文件直接报错提示先跑 src/script/train.py（2026-09-28 起缓存预建并入训练入口））；
本脚本不写
任何缓存文件，输出仅 partial h5ad 到 --out_dir。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 2026-09-28: tmp/ 下减一层

import hashlib
import h5py
import numpy as np
import pandas as pd
import torch
import tyro
import torchdiffeq
from scipy import sparse
from dataclasses import dataclass, field

from src.utils.config_utils import ConfigUtils, FlowConfig
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import make_lognorm_poisson_noise

ODEDEF_STEPS = 100  # 论文：Euler K=100 均匀步（附录 A.4.3）


@dataclass
class GenConfig(FlowConfig):
    checkpoint_path: str = ''   # main 模型 ckpt（2026-10-06：FlowConfig 字段删除后由子类自带，CLI 传）
    controls_dir: str = '/home/ict2/Projects/vcc-2026/resources/datasets/controls'  # context_{A,B,C}.h5ad + gene_names.csv + pert_counts.csv
    mask_fname: str = ''  # '' = 按训练同款公式派生（cache/vcc/mask_fold_...）
    out_dir: str = ''           # partial h5ad 输出目录
    shard_id: int = 0
    num_shards: int = 1
    batch_size: int = field(default=3, kw_only=True)  # ODE 批大小（2026-09-20 全轴 L=11,071：fp32 注意力显存墙 B≤3-4）
    seed: int = 42
    ode_steps: int = ODEDEF_STEPS
    top_infer_genes: int = field(default=19843, kw_only=True)  # 建模基因数（2026-09-30 定案=19,843 全轴；select_modeled_genes 内 min 到池大小）
    max_pairs: int = 0  # smoke: cap pairs per shard (0 = all)


def corpus_stem(config) -> str:
    """多文件语料（2026-09-27）：各文件名（去扩展名）排序后 + 拼接，与 data.py
    派生规则一致；2026-10-03：超 80 字符 sha1 缩短（与 data.py/_corpus_stem 同口径）。"""
    stem = '+'.join(os.path.splitext(os.path.basename(str(p)))[0]
                    for p in sorted(config.train_set_paths))
    if len(stem) > 80:
        import hashlib
        stem = 'h' + hashlib.sha1(stem.encode()).hexdigest()[:24]
    return stem


def artifact_paths(config):
    """与 data.py/process_vocab 完全一致的派生规则（只读）。"""
    stem = corpus_stem(config)
    pool_stem = (os.path.splitext(os.path.basename(str(config.train_pool_path)))[0]
                 if config.train_pool_path else 'all')
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'src')  # repo/src（2026-09-28 起本脚本在 tmp/，须补一层）
    cache = os.path.join(config.train_cache_dir, f'processed_n{config.n_top_genes}_{stem}_{pool_stem}.h5ad')
    mask_dir = config.train_cache_dir  # 2026-10-03：mask 与缓存同目录（用户定案）
    if config.mask_fname:
        mask = os.path.join(mask_dir, config.mask_fname)
    else:
        _neg = '_negative_edge' if config.use_negative_edge else ''
        mask = os.path.join(mask_dir,
                            f'mask_fold_{config.fold}topk_{config.topk}{config.split_method}{_neg}_{stem}_{pool_stem}.pt')
    vocab = os.path.join(src_dir, 'tokenizer',
                         f'{config.data_name}_{config.n_top_genes}_{stem}_{pool_stem}_highly_vocab.json')
    for p, what in [(cache + '.data.npy', 'processed cache'), (mask, 'coexpression mask'), (vocab, 'vocab')]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{what} missing: {p}\n'
                f'（训练/预构建会生成它：先跑 src/script/train.py 或 '
                f'python src/script/build_vcc_cache.py --data_name=vcc）')
    return cache, mask, vocab


def get_work_pairs(config: GenConfig):
    pairs = []
    for ctx in ['A', 'B', 'C']:
        pert_csv = os.path.join(config.controls_dir, 'pert_counts.csv')
        panel = pd.read_csv(pert_csv)['target_gene'].astype(str).tolist()
        for p in panel:
            pairs.append((ctx, p))
    pairs = pairs[config.shard_id::config.num_shards]
    if config.max_pairs:
        pairs = pairs[:config.max_pairs]
    return pairs


def stable_seed(ctx, pert, base):
    h = hashlib.md5(f'{ctx}:{pert}'.encode()).hexdigest()
    return (base + int(h[:8], 16)) % (2**31)


def read_var_names(f: h5py.File):
    var = f['var']
    if '_index' in var:
        ix = var['_index']
        if 'values' in ix:
            vals = ix['values'][:].astype(str)
            mask = ix['mask'][:] if 'mask' in ix else None
            return [v for v, m in zip(vals, mask)] if mask is not None else list(vals)
        return [x.decode() if isinstance(x, bytes) else str(x) for x in ix[:]]
    if 'gene_name' in var:
        g = var['gene_name']
        if 'categories' in g:
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g['categories'][:]]
            return [cats[c] for c in g['codes'][:]]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in g[:]]
    raise KeyError('could not read gene names from processed cache var')


def select_modeled_genes(cache: str, panel_csv_path: str, top_infer_genes: int,
                         vocab: GeneVocab, pool_path: str = '') -> list[str]:
    """从 processed cache 选建模基因（2026-09-21 定案：建模基因子集 = 完整基因轴，
    含 panel——panel 是其他扰动的真实 DEG，不可排除；推理侧仅对扰动自身靶列
    置 0，KD 语义）。口径：在采样池（pool_path，空串=整个基因轴）按
    dispersions_norm 取 top-N；2026-09-28 定案 top-N=18533 → 全部缓存列。"""
    with h5py.File(cache if os.path.exists(cache) else cache + '.meta.h5ad', 'r') as f:
        names = read_var_names(f)
        if 'dispersions_norm' in f['var']:
            disp = np.asarray(f['var']['dispersions_norm'][:])
        else:
            # 2026-10-03 流式缓存：HVG 跳过（全轴恒等），dispersions_norm 缺省；
            # top-N=19,843=全轴时排序无关 → 全 0 保持缓存列序
            disp = np.zeros(len(names), dtype=np.float64)
    name_set = set(names)
    pool: set | None = None
    if pool_path:
        pool = set(pd.read_csv(pool_path)['gene_name'].astype(str).tolist())
        pool &= name_set  # ∩ 语料 var
        assert pool, f'pool_path={pool_path!r} yields no usable genes'
    rank = np.argsort(-disp)
    cands = [names[i] for i in rank if (pool is None or names[i] in pool)]
    top = cands[:top_infer_genes]
    modeled = [g for g in top if g in vocab]
    return modeled


def _ode_forward(vf, gene_ids, x, t, src_b, pid_b, device,
                 gene_emb_c=None, value_emb_2_c=None, pert_emb_c=None):
    """模型前向包装：bf16 autocast（2026-10-04 回退，fp16 实测无提速——A800 fp16/bf16
    tensor core 同速率，profile 确认 GEMM 已走 ampere_bf16 核；bf16 指数 8 位更稳）。
    缓存项=t 无关量（ODE 步间复用）。"""
    t_ = t.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        return vf(gene_ids.repeat(x.shape[0], 1), x, t_, src_b, pid_b,
                  gene_ids.repeat(x.shape[0], 1),
                  gene_emb_cache=gene_emb_c, value_emb_2_cache=value_emb_2_c,
                  perturbation_emb_cache=pert_emb_c)


def ode_predict(vf, gene_ids, src_modeled, pert_id_b, batch_size, ode_steps,
                noise_type, poisson_alpha, poisson_target_sum, device, clamp_output=True,
                noise=None):
    """从控制谱生成扰动预测：噪声源与训练一致 → ODE t:0→1 → （可选 clamp≥0）。

    clamp_output：旧范式目标=log1p 表达（非负）需 clamp；残差目标范式（2026-09-26）
    目标=中心化 log2FC 残差（可负），必须 False。
    noise：固定初始噪声 (n_cells, L)（ODE 步数收敛性扫描用——同基因跨 N 共用，
    使解差只反映 Euler 截断误差）；None = 随机 randn（benchmark 常规路径）。"""
    L = gene_ids.shape[0]
    preds = []
    n_batches = (src_modeled.shape[0] + batch_size - 1) // batch_size
    with torch.no_grad():
        for s in range(0, src_modeled.shape[0], batch_size):
            _b0 = time.time()
            src_b = src_modeled[s:s + batch_size].contiguous()
            pid_b = pert_id_b.repeat(src_b.shape[0], 1)
            if noise is not None:
                nb = noise[s:s + batch_size].to(src_b.device)
            elif noise_type == 'Poisson':
                noise_ = make_lognorm_poisson_noise(
                    target_log=src_b, alpha=poisson_alpha, per_cell_L=poisson_target_sum)
                nb = noise_
            else:
                nb = torch.randn(src_b.shape[0], L, device=device)
            # 2026-09-19 加速：基因编码/对照值编码/扰动编码与 t 无关，在 ODE 的
            # 100 个 Euler 步之间完全重复——每批只算一次（与逐步重算同算子同
            # autocast，数值一致；仅推理路径，训练侧不受影响）。
            gid_b = gene_ids.repeat(src_b.shape[0], 1)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                gene_emb_c = vf.encoder(gid_b)
                value_emb_2_c = vf.value_encoder_2(src_b)
                pert_emb_c = vf.encoder(pid_b).mean(1)
            traj = torchdiffeq.odeint(
                lambda t, x: _ode_forward(vf, gene_ids, x, t, src_b, pid_b, device,
                                          gene_emb_c, value_emb_2_c, pert_emb_c),
                nb,
                torch.linspace(0, 1, ode_steps, device=device),
                atol=1e-4, rtol=1e-4, method='euler',
            )
            preds.append((torch.clamp(traj[-1], min=0) if clamp_output else traj[-1]).float())
            print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] ode: batch '
                  f'{s // batch_size + 1}/{n_batches} done '
                  f'({src_b.shape[0]} cells x {ode_steps} steps, {time.time() - _b0:.2f}s)',
                  flush=True)
    return torch.cat(preds, dim=0)


def log1p_bridge_to_counts(pred_modeled: np.ndarray, src_norm_full: np.ndarray,
                           src_depths: np.ndarray, modeled_idx: np.ndarray,
                           seed: int, zero_idx: np.ndarray | None = None) -> np.ndarray:
    """log1p 桥：建模基因<-模型、其余基因<-对照；expm1→按源细胞深度缩放→Poisson 计数。

    zero_idx（可选）：overwrite 之后、expm1/缩放之前置 0 的列——扰动靶基因（KD 语义，
    直接输出 0）。先置 0 再缩放 ⇒ 靶基因空出的 UMI 预算按组成性重新分配给其他基因
    （符合文库总量固定的真实测序语义），而不是让每个预测细胞总 UMI 缩水。"""
    full_log = src_norm_full.copy()
    full_log[:, modeled_idx] = pred_modeled
    if zero_idx is not None:
        full_log[:, zero_idx] = 0.0
    lam = np.expm1(np.clip(full_log, 0, 60))
    scale = src_depths / np.maximum(lam.sum(axis=1), 1.0)
    lam = lam * scale[:, None]
    counts = np.random.default_rng(seed).poisson(lam)
    return counts.astype(np.int32)


def main():
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)
    config = ConfigUtils.load(GenConfig, description=__doc__)
    assert config.checkpoint_path and os.path.exists(config.checkpoint_path)
    assert config.controls_dir and os.path.isdir(config.controls_dir)
    os.makedirs(config.out_dir, exist_ok=True)

    device = torch.device('cuda:0')
    torch.manual_seed(config.seed)

    # 1) 只读缓存/词表/mask（与训练同名派生；缺文件报错，不生成）
    cache, mask_path, vocab_path = artifact_paths(config)
    vocab = GeneVocab.from_file(vocab_path)
    modeled = select_modeled_genes(cache, config.panel_csv_path, config.top_infer_genes,
                                   vocab, pool_path=config.train_pool_path)
    print(f'modeled genes: {len(modeled)} (top-{config.top_infer_genes} by dispersion '
          f'within full axis − panel; panel targets zeroed at inference)', flush=True)

    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    L = len(modeled)

    # 2) model
    vf = instantiate_model(
        config.model_type, ntoken=config.ntoken, d_model=config.d_model,
        d_perturbation=config.d_model, fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function, mask_path=mask_path,
        num_latents=config.num_latents,  # perceiver_latent 用
    )
    ckpt = torch.load(config.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()

    # 3) official axis
    gene_names_csv = os.path.join(config.controls_dir, 'gene_names.csv')
    official_genes = pd.read_csv(gene_names_csv)['gene_name'].astype(str).tolist()
    modeled_idx = np.array([official_genes.index(g) for g in modeled], dtype=np.int64)

    # 4) generate per (context, pert)
    pairs = get_work_pairs(config)
    print(f'shard {config.shard_id}/{config.num_shards}: {len(pairs)} pairs', flush=True)

    out_rows, out_obs = [], []
    for ci, (ctx, pert) in enumerate(pairs):
        adata_path = os.path.join(config.controls_dir, f'context_{ctx}.h5ad')
        # controls loaded once per context and cached on this worker
        if not hasattr(main, '_ctl_cache') or main._ctl_cache[0] != ctx:
            import anndata as ad
            ctl = ad.read_h5ad(adata_path)
            raw = ctl.X.tocsr().astype(np.float32)
            # log1p space：normalize_total(1e4) + log1p，与训练一致
            norm = raw.copy()
            totals = np.asarray(norm.sum(axis=1)).ravel()
            norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
            norm.data = np.log1p(norm.data)
            main._ctl_cache = (ctx, raw, norm)
        _, raw, norm = main._ctl_cache

        rng = np.random.default_rng(stable_seed(ctx, pert, config.seed))
        src_idx = np.sort(rng.choice(raw.shape[0], 400, replace=False))

        src_raw = raw[src_idx]            # (400, 18533) 原始计数：深度 + 非建模基因来源
        src_norm = norm[src_idx]          # (400, 18533) log1p(CP10k) 对照
        depths = np.asarray(src_raw.sum(axis=1)).ravel()

        src_modeled = torch.from_numpy(src_norm[:, modeled_idx].toarray()).float().to(device)
        # 单槽扰动条件（方案B 2026-09-14）：encode 单个目标基因 → (B,1)，
        # 与训练/eval（run.py crisper 分支）对齐；勿再加 'control' 填充槽
        pert_id_b = torch.tensor(vocab.encode([pert]), dtype=torch.long, device=device).repeat(1, 1)

        pred_modeled = ode_predict(
            vf, gene_ids, src_modeled, pert_id_b, config.batch_size, config.ode_steps,
            config.noise_type, getattr(config, 'poisson_alpha', 0.8),
            getattr(config, 'poisson_target_sum', 1e4), device,
        ).cpu().numpy()  # (400, L) log1p 空间

        # 5) full vector + counts（log1p 桥）
        # 靶基因列置 0（KD 语义，2026-09-17 定案）：official panel ⊆ 官方轴
        counts = log1p_bridge_to_counts(pred_modeled, src_norm.toarray(), depths,
                                        modeled_idx, stable_seed(ctx, pert, config.seed + 7),
                                        zero_idx=np.array([official_genes.index(pert)], dtype=np.int64))

        out_rows.append(sparse.csr_matrix(counts))
        out_obs.append(pd.DataFrame({
            'target_gene': [pert] * 400,
            'context': [ctx] * 400,
        }))
        if (ci + 1) % 20 == 0:
            print(f'[{config.shard_id}] {ci+1}/{len(pairs)} done', flush=True)

    # 6) save partial h5ad
    import anndata as ad
    X = sparse.vstack(out_rows).tocsr()
    obs = pd.concat(out_obs, ignore_index=True)
    var = pd.DataFrame(index=official_genes)
    partial = ad.AnnData(X=X, obs=obs, var=var)
    out_path = os.path.join(config.out_dir, f'partial_s{config.shard_id}.h5ad')
    partial.write_h5ad(out_path)
    print(f'wrote {partial.shape} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
