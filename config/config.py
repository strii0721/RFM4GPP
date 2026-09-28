
from dataclasses import dataclass, field
import os
from datetime import datetime


@dataclass
class CommonConfig:
    """公共路径配置（2026-09-27 用户定案：原 config_flow.py 模块级常量迁入此类）。

    训练集不再硬编码任何数据集（不限于 replogle+nadig），语料路径一律经
    --corpus_paths 显式传入；本类仅承载 VCC-2026 竞赛固定基础设施
    （官方合并语料、controls 目录、官方 300 panel 清单）与资源根目录。
    """
    resource_root: str = '/home/ict2/Projects/vcc-2026/resources/datasets'
    # 官方合并语料（train_merged；历史 gen 提交/分析用，非训练默认语料）
    corpus_path: str = os.path.join(resource_root, 'train_merged', 'train_merged_panel.h5ad')
    # controls 目录：context_{A,B,C}.h5ad + gene_names.csv + pert_counts.csv
    controls_dir: str = os.path.join(resource_root, 'controls')
    # VCC-2026 官方 300 panel 清单（竞赛固定，与训练语料无关）
    panel_path: str = os.path.join(controls_dir, 'pert_counts.csv')


@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'

    # Flow Matching specific parameters（默认=论文附录 A.4.3 口径）
    batch_size: int = 96          # 论文全局 batch=96；8 卡 DDP 时 train.sh 按 96/GPUS 分摊到每 rank
    # replogle 全轴建模（2026-09-17 用户定案）：缓存保留全部 11,919 基因列，
    # vocab = 11,919 + 4 specials = 11,923。instantiate_model 已透传 ntoken/d_model
    # （此前恒 6000 是死参数，全轴 vocab 会索引越界）。
    ntoken: int = 11923
    d_model: int = 512
    lr: float = 5e-5              # 论文 Adam lr=5e-5 余弦衰减
    steps: int = 10000           # 默认 10k（2026-09-26 用户定案：残差范式 ~10k 收敛）；启动可 --steps 覆盖
    eta_min: float = 1e-6         # 论文衰减下界 ηmin=1e-6
    devices: str = "1"
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "vcc"
    perturbation_function: str = 'crisper'
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    print_every: int = 1000
    mode: str = 'predict_y' # predict_y, predict_p
    result_path: str = 'output/train'
    perturbation_fusion_method: str = 'sum' # mlp, sum
    fusion_method: str = 'differential_perceiver' # cross , concat, add
    infer_top_gene: int = 11919
    # 2026-09-21 用户定案：训练每步从【完整基因轴】（含 300 panel，固定集合）随机
    # 抽全轴（min 上限 = 缓存列 11,371）。panel 是其他扰动的真实 DEG 不可排除；
    # 推理侧仅对扰动自身靶列置 0。train_pool_path='' 时 data.py 用 HVG
    # n_top_genes=11919 实质保留全部有 dispersion 的列（零方差 548 列剔除，
    # 实测 11,371），缓存 ~87GB。
    n_top_genes: int = 11919
    checkpoint_path: str = ''
    gamma: float = 0.5             # 论文 MMD λ=0.5
    # replogle 留系（2026-09-17 用户定案）：train 文件（K562/Jurkat/HepG2）全量训练，
    # 无内部留出；测试语料 = 独立文件 test_corpus_path（RPE1），benchmark 专用。
    # 原方案仍可用 --split_method=single_line（heldout_line 系留出）或 single（基因留出）。
    split_method: str = 'whole'
    use_mmd_loss: bool = False     # 残差目标范式去 MMD（2026-09-26 用户定案；原论文带动态多核 RBF）
    fold: int = 0
    use_negative_edge: bool = True # 论文 kNN k=30 带符号相关（signed mask）
    topk: int = 30                 # 论文 k=30

    # VCC-2026 mode (data_name='vcc')
    # data_path = 缓存/共表达图/split 产物根目录。2026-09-20 用户定案：项目整体
    # 迁至家目录 /home/ict2/Projects（软链接 → /share/ict2/Projects 共享盘 20T），
    # 缓存/数据集全部随迁；训练语料走 corpus_paths 显式传入。
    data_path: str = '/home/ict2/Projects/scDFM/tmp'
    # 训练语料文件列表（2026-09-27 用户定案）：支持分散在多个文件夹的多个文件，
    # 建缓存时按序读入并沿 obs 拼接（var 轴须逐文件一致）。默认空=必须显式传入，
    # 不再绑定任何数据集。CLI：--corpus_paths <p1> <p2> ...（空格分隔，遇下一 --flag 停）。
    corpus_paths: list[str] = field(default_factory=list)
    panel_path: str = CommonConfig().panel_path  # VCC-2026 官方 300 panel 清单（竞赛固定，与训练语料无关）
    test_corpus_path: str = '/home/ict2/Projects/vcc-2026/resources/datasets/replogle/replogle_rpe1.h5ad'  # 竞赛留系真实侧（RPE1），benchmark 专用
    train_pool_path: str = ''  # 训练每步采样池（非空=基因清单；空串=整个基因轴，含 panel，2026-09-21 定案）
    line_col: str = 'context'   # replogle train 文件 context=K562/Jurkat/HepG2（obs 无 cell_line 列）
    max_len: int = 500
    max_len_batch: int = 1000
    heldout_line: str = 'RPE1'  # single_line 留系口径（RPE1）；whole 口径下被 test_corpus_path 取代
    # 残差目标范式（2026-09-26 用户定案）：冻结三张量目录（rbar_c/rbar_p/gbar/res_*）
    residual_targets_dir: str = 'tmp/residual_targets'
    condition_token_ratio: float = 0.25  # 条件单元格采样比例（per batch）
    condition_max_tokens: int = 400      # 条件单元格 token 上限（Perceiver 输入）
    mask_subsample: int = 50000  # cells for co-expression graph (0 = all)
    max_test_perts: int = 20  # cap on perturbations evaluated per checkpoint (0 = all)
    num_workers: int = 4  # DataLoader workers per rank
    use_bf16: bool = True  # bf16 autocast for forward/backward
    do_eval: bool = False  # in-loop eval; MUST be False under multi-GPU DDP (deadlocks NCCL)
    # same-line pairing (TrainSampler): a (line, gene) pool must have at least this
    # many cells for the line to be eligible; measured 2026-09-11 on the merged
    # corpus (iscr12g+xatlas+replogle): >=20 keeps ~2,011 eligible combos
    min_tgt_cells: int = 20

    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()

    def make_path(self):
        # timestamp IS the experiment name: output/train_{YYYY-MM-DD_HH-MM}/
        # SCDFM_RUN_TS（train.sh 启动时刻导出）保证与 logs/train_{ts} 同名对应
        ts = os.environ.get('SCDFM_RUN_TS') or datetime.now().strftime('%Y-%m-%d_%H-%M')
        return os.path.join(os.path.dirname(self.result_path),
                            os.path.basename(self.result_path) + '_' + ts)
