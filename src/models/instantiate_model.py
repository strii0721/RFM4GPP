from src.models.scGPT.model import TransformerModel
from src.models.perturbation.model import Model as FlowModel
from src.models.perturbation.model import TimedTransformer
from src.models.origin.model import model as OriginModel
import torch

def instantiate_model(model_type: str, **kwargs):

    if model_type == 'origin':
        if kwargs['fusion_method'] == 'differential_transformer':
            layers = 8
        elif kwargs['fusion_method'] == 'differential_perceiver':
            layers = 4
        elif kwargs['fusion_method'] == 'perceiver_latent':
            layers = 4  # 与 differential_perceiver 同层数（2026-10-05，便于对照归因）
        else:
            layers = 8
        # ntoken/d_model 必须透传（2026-09-17）：此前恒用 OriginModel 默认
        # ntoken=6000，全轴 replogle vocab（11,923）会索引越界。
        return OriginModel(ntoken=kwargs['ntoken'], d_model=kwargs['d_model'],
                           fusion_method=kwargs['fusion_method'], nlayers=layers,
                           perturbation_function=kwargs['perturbation_function'],
                           mask_path=kwargs['mask_path'],
                           use_perturbation_interaction=kwargs.get('use_perturbation_interaction', True),
                           pert_ntoken=kwargs.get('pert_ntoken'))
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
if __name__ == "__main__":
    model = instantiate_model("punet128")
    x = torch.randn(32,  128, 128)
    t = torch.randn(32)
    out = model( x,t)
    print(out.shape)