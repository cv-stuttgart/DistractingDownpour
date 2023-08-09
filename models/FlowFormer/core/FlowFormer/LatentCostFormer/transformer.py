import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from models.FlowFormer.core.utils.utils import coords_grid, bilinear_sampler, upflow8
from models.FlowFormer.core.FlowFormer.common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from models.FlowFormer.core.FlowFormer.encoders import twins_svt_large_context, twins_svt_large
from models.FlowFormer.core.position_encoding import PositionEncodingSine, LinearPositionEncoding
from models.FlowFormer.core.FlowFormer.LatentCostFormer.twins import PosConv
from models.FlowFormer.core.FlowFormer.LatentCostFormer.encoder import MemoryEncoder
from models.FlowFormer.core.FlowFormer.LatentCostFormer.decoder import MemoryDecoder
from models.FlowFormer.core.FlowFormer.LatentCostFormer.cnn import BasicEncoder

class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, image1, image2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        return flow_predictions
