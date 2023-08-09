import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, kaiming_normal_

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn(
            "failed to load custom correlation module" "which is needed for FlowNetC",
            ImportWarning,
        )


def conv(
    batchNorm: bool,
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride=1,
    dilation: int = 1,
):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=((kernel_size - 1) // 2) * dilation,
                bias=True,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def correlate(input1, input2, return_feat_maps=False):
    out_corr = spatial_correlation_sample(
        input1,
        input2,
        kernel_size=1,
        patch_size=21,
        stride=1,
        padding=0,
        dilation_patch=2,
    )
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    if return_feat_maps:
        return F.leaky_relu_(out_corr, 0.1), out_corr
    else:
        return F.leaky_relu_(out_corr, 0.1)


class FlowNetC_flexible_larger_field(nn.Module):
    expansion = 1

    def __init__(
        self,
        batchNorm: bool = False,
        div_flow: int = 1,
        kernel_size: int = 5,
        number_of_reps=1,
        dilation: int = 1,
        return_feat_maps: bool = False,
    ):
        super().__init__()

        self.div_flow = div_flow
        self.kernel_size = kernel_size
        self.number_of_reps = number_of_reps
        self.dilation = dilation
        self.return_feat_maps = return_feat_maps

        self.batchNorm = batchNorm
        self.convs1 = nn.ModuleList()
        self.convs1.append(
            conv(
                self.batchNorm,
                3,
                64,
                kernel_size=7,
                stride=2,
                dilation=self.dilation,
            )
        )
        for _ in range(self.number_of_reps):
            self.convs1.append(
                conv(
                    self.batchNorm,
                    64,
                    64,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )

        self.convs2 = nn.ModuleList()
        self.convs2.append(
            conv(
                self.batchNorm,
                64,
                128,
                kernel_size=self.kernel_size,
                stride=2,
                dilation=self.dilation,
            )
        )
        for _ in range(self.number_of_reps):
            self.convs2.append(
                conv(
                    self.batchNorm,
                    128,
                    128,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )

        self.convs3 = nn.ModuleList()
        self.convs3.append(
            conv(
                self.batchNorm,
                128,
                256,
                kernel_size=self.kernel_size,
                stride=2,
                dilation=self.dilation,
            )
        )
        for _ in range(self.number_of_reps):
            self.convs3.append(
                conv(
                    self.batchNorm,
                    256,
                    256,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )

        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.upsample1 = torch.nn.Upsample(scale_factor=4, mode="bilinear")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        self.convs1.apply(self._init_weights)
        self.convs2.apply(self._init_weights)
        self.convs3.apply(self._init_weights)

    def _init_weights(self, m):  # pylint: disable=no-self-use
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            kaiming_normal_(m.weight, 0.1)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)

    def normalize(self, im):  # pylint: disable=no-self-use
        mean = np.array([0.40066648, 0.39482617, 0.3784785])  # RGB
        # mean=np.array([0.3784785,0.39482617,0.40066648]) # BGR
        std = np.array([1, 1, 1])
        return (
            im - torch.from_numpy(mean[None, :, None, None]).cuda()
        ) / torch.from_numpy(std[None, :, None, None]).cuda()

    def forward(self, x1, x2):
        x1 = self.normalize(x1).float()
        x2 = self.normalize(x2).float()

        if self.return_feat_maps:
            return_feat_maps = []

        out_conv1a = x1
        for conv in self.convs1:
            out_conv1a = conv(out_conv1a)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv1a.clone())

        out_conv2a = out_conv1a
        for conv in self.convs2:
            out_conv2a = conv(out_conv2a)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv2a.clone())

        out_conv3a = out_conv2a
        for conv in self.convs3:
            out_conv3a = conv(out_conv3a)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv3a.clone())

        out_conv1b = x2
        for conv in self.convs1:
            out_conv1b = conv(out_conv1b)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv1b.clone())

        out_conv2b = out_conv1b
        for conv in self.convs2:
            out_conv2b = conv(out_conv2b)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv2b.clone())

        out_conv3b = out_conv2b
        for conv in self.convs3:
            out_conv3b = conv(out_conv3b)
        if self.return_feat_maps:
            return_feat_maps.append(out_conv3b.clone())

        out_conv_redir = self.conv_redir(out_conv3a)

        if self.return_feat_maps:
            out_correlation, out_corr_before_act = correlate(
                out_conv3a, out_conv3b, return_feat_maps=True
            )
            return_feat_maps.append(out_corr_before_act.clone())
        else:
            out_correlation = correlate(out_conv3a, out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            if self.return_feat_maps:
                return self.upsample1(flow2 * self.div_flow), return_feat_maps
            else:
                return self.upsample1(flow2 * self.div_flow)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]
