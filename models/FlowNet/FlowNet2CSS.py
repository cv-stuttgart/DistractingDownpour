import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

import sys
import os

from models.FlowNet.resample2d_package.resample2d import Resample2d
from models.FlowNet.channelnorm_package.channelnorm import ChannelNorm

from models.FlowNet import FlowNetC
from models.FlowNet import FlowNetS

from models.FlowNet.submodules import *

class FlowNet2CSS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CSS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample2 = Resample2d()


        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest') 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow