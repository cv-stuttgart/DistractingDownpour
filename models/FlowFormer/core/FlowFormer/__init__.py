import torch
def build_flowformer(cfg):
    name = cfg.transformer 
    if name == 'latentcostformer':
        from models.FlowFormer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")

    return FlowFormer(cfg[name])