"""配置读取工具（2026-09-28 用户定案）。

- 一切配置值从 configs/universal.yaml 读入：CommonConfig / FlowConfig 不再携带
  默认值，YAML 缺 key 直接报错（required 字段缺失），无静默兜底。
- 优先级：CLI 显式参数（tyro 以 Optional schema 解析，只取显式传入的 flag）> YAML。
- 节语义：
    common 节 —— 对所有配置类生效（语料列表 / panel CSV / 冻结张量目录）。
    flow 节 —— 对 FlowConfig 及其子类生效；子类自身重定义的字段（如 batch_size）
               不被 YAML 覆盖，走子类默认 + CLI。
- 子类（GenConfig/BenchConfig/ScanConfig）自身的运行态字段保留各自默认值，
  不在 universal.yaml 中配置（per-task 参数走 CLI）。
"""
import dataclasses
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import tyro
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIVERSAL_YAML = os.path.join(_PROJECT_ROOT, 'configs', 'universal.yaml')


@dataclass
class CommonConfig:
    """公共路径配置（无默认值；全部从 universal.yaml common 节读入）。"""
    train_set_paths: list[str]
    test_set_paths: str
    inference_control_paths: list[str]  # 推理 control 多源（2026-09-30 用户定案）
    cache_workers: int   # 缓存流式构建的并行 worker 数（2026-09-30：文件级并行，≤文件数）
    panel_csv_path: str
    frozen_tensors_dir: str
    train_cache_dir: str
    # 扰动方向白名单（2026-09-28 用户定案）：训练侧只保留 exo_perturb_subtype 值
    # 在此列表内的细胞。当前 [CRISPRi]=只留敲低；空列表=不过滤。
    perturb_direction: list[str]
    # obs 语义列匹配表（2026-09-28 用户定案）：不同数据集构建方表头命名相互独立，
    # 每个语义字段按候选列表顺序取第一个匹配上的列（derive_pert_columns）。
    obs_col_candidates: dict
    # 扰动基因列中的对照哨兵值（无 ctrl_flag 候选列时用于判定对照）。
    ctrl_sentinels: list[str]
    # checkpoint 基路径（2026-09-28 定案：train.py 在此基名下拼 _<ts>/iteration_N）
    # 输出基目录（train checkpoint 与 inference 产物共用，2026-09-29 合并定案；
    # 代码拼 <基目录>/<任务名>_<ts>，如 output/train_<ts>、output/inference_<ts>）
    output_base_dir: str
    # 日志基目录（train/build_tensors/inference 三入口共用）
    log_base_dir: str
    # real/pred 细胞数口径（2026-09-28 从 inference 节提取到 common：build_tensors
    # real 侧三件套与 inference 共用；官方 context=18400 取子集控时长）
    n_ctrl_cells: int       # real 侧对照细胞数（ODE 源 / DE 参考组）
    n_pred_cells: int       # 每扰动预测细胞数（官方 400）
    n_real_cells: int       # real 侧每扰动参考细胞数上限（不足用全部真实细胞）
    min_real_cells: int     # real 侧每扰动最少细胞数（低于则跳过该基因）


@dataclass
class FlowConfig:
    """训练/模型配置（无默认值；全部从 universal.yaml flow 节读入）。"""
    # Flow model type
    model_type: str

    # Flow Matching specific parameters（默认=论文附录 A.4.3 口径）
    batch_size: int          # 论文全局 batch=96；8 卡 DDP 时 train.py 按 batch_total/gpus 分摊到每 rank
    ntoken: int              # 11,919 全轴基因 + 4 specials = 11,923
    d_model: int
    lr: float                # 论文 Adam lr=5e-5 余弦衰减
    steps: int               # 2026-09-26 定案：残差范式 ~10k 收敛；--steps 可覆盖
    eta_min: float           # 论文衰减下界 ηmin=1e-6
    devices: str
    test_only: bool
    # Perturbation related parameters
    data_name: str
    perturbation_function: str
    noise_type: str
    poisson_alpha: float
    poisson_target_sum: int

    print_every: int
    mode: str                # predict_y, predict_p
    perturbation_fusion_method: str  # mlp, sum
    fusion_method: str       # cross , concat, add
    infer_top_gene: int
    # 2026-09-21 用户定案：训练每步从【完整基因轴】（含 300 panel，固定集合）随机
    # 抽全轴（min 上限 = 缓存列 11,371）。panel 是其他扰动的真实 DEG 不可排除；
    # 推理侧仅对扰动自身靶列置 0。train_pool_path='' 时 data.py 用 HVG
    # n_top_genes=11919 实质保留全部有 dispersion 的列（零方差 548 列剔除，
    # 实测 11,371），缓存 ~87GB。
    n_top_genes: int
    checkpoint_path: str
    gamma: float             # 论文 MMD λ=0.5
    # replogle 留系（2026-09-17 用户定案）：train 文件（K562/Jurkat/HepG2）全量训练，
    # 无内部留出；测试语料 = 独立文件 test_set_paths（RPE1），benchmark 专用。
    # 原方案仍可用 --split_method=single_line（heldout_line 系留出）或 single（基因留出）。
    split_method: str
    use_mmd_loss: bool       # 残差目标范式去 MMD（2026-09-26 用户定案；原论文带动态多核 RBF）
    fold: int
    use_negative_edge: bool  # 论文 kNN k=30 带符号相关（signed mask）
    topk: int                # 论文 k=30

    # VCC-2026 mode (data_name='vcc')
    # data_path = 缓存/共表达图/split 产物根目录。2026-09-20 用户定案：项目整体
    # 迁至家目录 /home/ict2/Projects（软链接 → /share/ict2/Projects 共享盘 20T）。
    data_path: str
    batch_total: int          # 全局 batch（train.py 按 gpus 分摊到每 rank）
    gpus: int                 # DDP 卡数（torchrun --nproc_per_node 须一致）
    # 训练语料文件列表：位置在 universal.yaml common 节配置（2026-09-28 用户定案，
    # 启动勿需 --train_set_paths flag；tyro 仍可覆盖）。建缓存时按序读入并沿 obs 拼接
    # （var 轴须逐文件一致）。
    train_set_paths: list[str]
    panel_csv_path: str          # VCC-2026 官方 300 panel 清单（竞赛固定，与训练语料无关）
    test_set_paths: str    # 竞赛留系真实侧（RPE1），benchmark 专用
    inference_control_paths: list[str]  # 推理 control 多源（2026-09-30 用户定案；common 节）
    cache_workers: int   # 缓存流式构建并行 worker 数（2026-09-30；common 节）
    train_pool_path: str     # 训练每步采样池（非空=基因清单；空串=整个基因轴，含 panel，2026-09-21 定案）
    line_col: str            # replogle train 文件 context=K562/Jurkat/HepG2（obs 无 cell_line 列）
    max_len: int
    max_len_batch: int
    heldout_line: str        # single_line 留系口径（RPE1）；whole 口径下被 test_set_paths 取代
    # 残差目标范式（2026-09-26 用户定案）：冻结三张量目录（rbar_c/rbar_p/gbar/res_*）
    # 位置在 universal.yaml common 节配置（2026-09-28）
    frozen_tensors_dir: str
    train_cache_dir: str      # 训练缓存目录（processed_*.h5ad/.meta；common 节配置）
    # 扰动方向白名单（common 节；2026-09-28 定案）：训练侧只保留 exo_perturb_subtype
    # 值在此列表内的细胞。当前 [CRISPRi]=只留敲低；空列表=不过滤。
    perturb_direction: list[str]
    # obs 语义列匹配表 + 对照哨兵值（common 节；2026-09-28 定案，derive_pert_columns 用）
    obs_col_candidates: dict
    ctrl_sentinels: list[str]
    # 输出基目录 + 日志基目录（common 节；2026-09-29 合并定案）
    output_base_dir: str
    log_base_dir: str
    # real/pred 细胞数口径（common 节；2026-09-28 从 inference 节提取，
    # build_tensors real 侧三件套与 inference 共用）
    n_ctrl_cells: int
    n_pred_cells: int
    n_real_cells: int
    min_real_cells: int
    condition_token_ratio: float  # 条件单元格采样比例（per batch）
    condition_max_tokens: int     # 条件单元格 token 上限（Perceiver 输入）
    mask_subsample: int       # cells for co-expression graph (0 = all)
    max_test_perts: int       # cap on perturbations evaluated per checkpoint (0 = all)
    num_workers: int          # DataLoader workers per rank
    use_bf16: bool            # bf16 autocast for forward/backward
    do_eval: bool             # in-loop eval; MUST be False under multi-GPU DDP (deadlocks NCCL)
    # same-line pairing (TrainSampler): a (line, gene) pool must have at least this
    # many cells for the line to be eligible; measured 2026-09-11 on the merged
    # corpus (iscr12g+xatlas+replogle): >=20 keeps ~2,011 eligible combos
    min_tgt_cells: int
    # CRISPR 过滤开关（2026-09-28 补回：config_flow.py→yaml 迁移时丢失，缓存重建必崩）。
    # 已被 common.perturb_direction 取代（列表白名单，见 CommonConfig），保留仅为兼容。
    crispr_type_col: str
    crispr_type_value: str

    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()

    def make_path(self, task: str = 'train'):
        # timestamp IS the experiment name: <output_base_dir>/<task>_{YYYY-MM-DD_HH-MM}/
        # SCDFM_RUN_TS（train.py 启动时刻设定）保证与 logs/<task>_{ts} 同名对应
        ts = os.environ.get('SCDFM_RUN_TS') or datetime.now().strftime('%Y-%m-%d_%H-%M')
        return os.path.join(self.output_base_dir, f'{task}_{ts}')


class ConfigUtils:
    """配置读取入口。用法：

        from src.utils.config_utils import ConfigUtils, FlowConfig
        cfg = ConfigUtils.load(FlowConfig, description=__doc__)
    """

    @staticmethod
    def load_yaml(yaml_path: Optional[str] = None) -> dict:
        with open(yaml_path or UNIVERSAL_YAML, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def load(cls, config_cls, description=None, yaml_path=None, use_cli=True,
             extra_sections: tuple = ()):
        """读 universal.yaml 构造 config_cls；CLI 显式参数覆盖 YAML。

        use_cli=False：纯 YAML 读取（脚本用 argparse 自行解析命令行时，如
        build_tensors），tyro 不参与。
        extra_sections：额外合并的 YAML 节（在 common/flow 之后，优先级更高）。
        如 inference 节供 BenchConfig（推理运行态参数，2026-09-28 用户定案）。
        """
        raw = cls.load_yaml(yaml_path)
        section = dict(raw.get('common', {}))
        if issubclass(config_cls, FlowConfig):
            flow = dict(raw.get('flow', {}))
            for k in cls._subclass_override_fields(config_cls):
                flow.pop(k, None)
            section.update(flow)
        for name in extra_sections:
            section.update(dict(raw.get(name, {}) or {}))
        cfg = config_cls(**section)  # YAML 缺 key → TypeError（无默认值兜底，用户定案）
        if use_cli:
            cls._apply_cli(cfg, config_cls, description)
        return cfg

    @staticmethod
    def _subclass_override_fields(config_cls) -> set:
        """config_cls 相对 FlowConfig 新增/重定义的字段（不被 YAML flow 节覆盖）。

        注意：dataclass 装饰器在子类 __dataclass_fields__ 里存的是含继承的全量
        字段表，必须做集合差（直接遍历 __dict__ 会把 base 字段也卷进来）。
        """
        if config_cls is FlowConfig:
            return set()
        # 子类自身声明的字段（__annotations__ 只含本类声明，含重定义字段如 batch_size；
        # 新增字段不在 flow 节里，pop 为无害 no-op）
        return set(config_cls.__dict__.get('__annotations__', {}))

    @staticmethod
    def _apply_cli(cfg, config_cls, description):
        """tyro 解析 CLI schema：非 bool 字段 default None（仅显式传入取到值）；
        bool 字段 default False（--flag/--no-flag），仅在 argv 显式出现时覆盖
        （本版本 tyro 不支持 bool+None 默认的裸 flag 风格）。"""
        fields = []
        bool_names = []
        for f in dataclasses.fields(config_cls):
            if f.type is bool:
                fields.append((f.name, bool, dataclasses.field(default=False)))
                bool_names.append(f.name)
            else:
                fields.append((f.name, f.type, dataclasses.field(default=None)))
        cli_cls = dataclasses.make_dataclass(config_cls.__name__ + '_CLI', fields)
        if description is not None:
            parsed = tyro.cli(cli_cls, description=description)
        else:
            parsed = tyro.cli(cli_cls)
        argv = sys.argv[1:]
        # 归一化连字符（--reuse_real 与 --reuse-real 等价；2026-09-29 修复：
        # spawn 传下划线形式时门控不匹配 → bool flag 静默失效，worker 不复用
        # real.h5ad、各自并发写同一文件触发 HDF5 锁冲突）
        argv_norm = [a.replace('_', '-') for a in argv if a.startswith('--')]
        for f in dataclasses.fields(config_cls):
            v = getattr(parsed, f.name)
            if f.name in bool_names:
                hyphen = f.name.replace('_', '-')
                if f'--{hyphen}' in argv_norm or f'--no-{hyphen}' in argv_norm:
                    setattr(cfg, f.name, v)
            elif v is not None:
                setattr(cfg, f.name, v)
