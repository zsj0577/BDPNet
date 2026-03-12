from __future__ import print_function, division
from .DWT_IDWT_layer import DWT_2D

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


norm_op_kwargs = {'eps': 1e-5, 'affine': True}
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}

class BDPNet(nn.Module):
    """HMRNet-p is designed for plump anatomical structures
             which contains 3x3x3 convolutional blocks."""

    def __init__(self, params):
        super(BDPNet, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.fr_chs = self.params['fr_feature_chns']
        self.num_classes = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.conv_op = self.params['con_op']
        self._deep_supervision = self.params['_deep_supervision']
        self.do_ds = self.params['do_ds']
        filters = [32, 64, 128, 256]

        assert (len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0])
        self.fr_in_conv = ConvBlock(self.in_chns, self.fr_chs)

        self.BFI1 = BFI(filters[0], filters[0], 3, 8, 2)
        self.BFI11 = BFI(filters[0] // 4, filters[0], 3, 8, 2)
        self.BFI2 = BFI(filters[1], filters[0], 3, 8, 2)
        self.BFI3 = BFI(filters[2], filters[0], 3, 8, 2)
        self.BFI_d3 = BFI(filters[0] // 2, filters[0], 3, 8, 2)
        self.BFI_d2 = BFI(filters[0], filters[0], 3, 8, 2)
        self.BFI_d1 = BFI(filters[1], filters[0], 3, 8, 2)

        self.basic_conv1 = MSFE(self.fr_chs, self.fr_chs)  # full _resolution conv
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1])  # Unet  feature down sample
        self.basic_down1 = DownBlock_Con(pooling_p=2)  # full_resolution down sample
        self.basic_up1 = UpBlock_Con(scale_factor=2)  # Unet  feature up sample
        self.concat1 = Concatenate(self.ft_chns[1] + self.fr_chs, self.ft_chns[1])
        self.fr_concat1 = Concatenate(self.ft_chns[1] + self.fr_chs, self.fr_chs)

        self.basic_conv2 = MSFE(self.fr_chs, self.fr_chs)
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2])
        self.basic_down2 = DownBlock_Con(pooling_p=4)
        self.basic_up2 = UpBlock_Con(scale_factor=4)
        self.concat2 = Concatenate(self.ft_chns[2] + self.fr_chs, self.ft_chns[2])
        self.fr_concat2 = Concatenate(self.ft_chns[2] + self.fr_chs, self.fr_chs)

        self.basic_conv3 = MSFE(self.fr_chs, self.fr_chs)
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3])
        self.basic_down3 = DownBlock_Con(pooling_p=8)
        self.basic_up3 = UpBlock_Con(scale_factor=8)
        self.dpvision_con3 = nn.Conv2d(self.ft_chns[3], self.num_classes, kernel_size=1)  # deep supervison
        self.concat3 = Concatenate(self.ft_chns[3] + self.fr_chs, self.ft_chns[3])
        self.fr_concat3 = Concatenate(self.ft_chns[3] + self.fr_chs, self.fr_chs)

        if (len(self.ft_chns) == 5):
            self.basic_conv4 = MSFE(self.fr_chs, self.fr_chs)
            self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4])
            self.basic_down4 = DownBlock_Con(self.fr_chs, self.ft_chns[4], pooling_p=16)
            self.basic_up4 = UpBlock_Con(self.ft_chns[4], self.fr_chs, scale_factor=16)
            self.concat4 = Concatenate(self.ft_chns[4])
            self.fr_concat4 = Concatenate(self.fr_chs)

            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],
                               bilinear=self.bilinear)

        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2],
                           bilinear=self.bilinear)
        self.basic_conv5 = MSFE(self.fr_chs, self.fr_chs)
        self.basic_down5 = DownBlock_Con(pooling_p=4)
        self.basic_up5 = UpBlock_Con(scale_factor=4)
        self.dpvision_con5 = nn.Conv2d(self.ft_chns[2], self.num_classes, kernel_size=1)  # deep supervison
        self.concat5 = Concatenate_Threechs(2 * self.ft_chns[2] + self.fr_chs, self.ft_chns[2])
        self.fr_concat5 = Concatenate(self.ft_chns[2] + self.fr_chs, self.fr_chs)

        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1],
                           bilinear=self.bilinear)
        self.basic_conv6 = MSFE(self.fr_chs, self.fr_chs)
        self.basic_down6 = DownBlock_Con(pooling_p=2)
        self.basic_up6 = UpBlock_Con(scale_factor=2)
        self.dpvision_con6 = nn.Conv2d(self.ft_chns[1], self.num_classes, kernel_size=1)  # deep supervison
        self.concat6 = Concatenate_Threechs(2 * self.ft_chns[1] + self.fr_chs, self.ft_chns[1])
        self.fr_concat6 = Concatenate(self.ft_chns[1] + self.fr_chs, self.fr_chs)

        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0],
                           bilinear=self.bilinear)
        self.basic_conv7 = MSFE(self.fr_chs, self.fr_chs)
        self.concat7 = Concatenate_Threechs(2 * self.ft_chns[0] + self.fr_chs, self.ft_chns[0])
        self.fr_concat7 = Concatenate(self.ft_chns[0] + self.fr_chs, self.fr_chs)
        self.final_concat = Concatenate(self.ft_chns[0] + self.fr_chs, self.ft_chns[0])

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.num_classes,
                                  kernel_size=3, padding=1)
        self.softmax = lambda x: F.softmax(x, 1)

    def forward(self, x):
        segout = []
        x0 = self.in_conv(x)
        fr_x0 = self.fr_in_conv(x)

        x1 = self.down1(x0)  # conv  + down sample
        x11 = self.basic_up1(x1)  #up sample
        fr_x1 = self.basic_conv1(fr_x0)  # conv
        fr_x11 = self.basic_down1(fr_x1)  # maxpool
        x1 = self.concat1(x1, fr_x11)
        x1 = self.BFI1(x1)
        fr_x1 = self.fr_concat1(fr_x1, x11)
        fr_x1 = self.BFI11(fr_x1)
        x2 = self.down2(x1)
        x22 = self.basic_up2(x2)
        fr_x2 = self.basic_conv2(fr_x1)
        fr_x22 = self.basic_down2(fr_x2)
        x2 = self.concat2(x2, fr_x22)
        x2 = self.BFI2(x2)
        fr_x2 = self.fr_concat2(fr_x2, x22)
        fr_x2 = self.BFI11(fr_x2)

        x3 = self.down3(x2)
        x33 = self.basic_up3(x3)
        fr_x3 = self.basic_conv3(fr_x2)
        fr_x33 = self.basic_down3(fr_x3)
        x3 = self.concat3(x3, fr_x33)
        x3 = self.BFI3(x3)
        deep_x3 = self.dpvision_con3(x3)
        segout.append(deep_x3)
        fr_x3 = self.fr_concat3(fr_x3, x33)
        fr_x3 = self.BFI11(fr_x3)

        if (len(self.ft_chns) == 5):
            x4 = self.down4(x3)
            x44 = self.basic_up4(x4)
            fr_x4 = self.basic_conv4(fr_x3)
            fr_x44 = self.basic_down4(fr_x4)
            x4 = self.concat4(x4, fr_x44)
            fr_x4 = self.fr_concat4(fr_x4, x44)

            x = self.up1(x4, x3)

        else:
            x = x3
            fr_x = fr_x3

        x5 = self.up2(x)
        x55 = self.basic_up5(x5)
        fr_x5 = self.basic_conv5(fr_x)
        fr_x55 = self.basic_down5(fr_x5)
        x5 = self.concat5(x5, fr_x55, x2)
        x5 = self.BFI_d1(x5)
        deep_x5 = self.dpvision_con5(x5)
        segout.append(deep_x5)
        fr_x5 = self.fr_concat5(fr_x5, x55)
        fr_x5 = self.BFI11(fr_x5)

        x6 = self.up3(x5)
        x66 = self.basic_up6(x6)
        fr_x6 = self.basic_conv6(fr_x5)
        fr_x66 = self.basic_down6(fr_x6)
        x6 = self.concat6(x6, fr_x66, x1)
        x6 = self.BFI_d2(x6)
        deep_x6 = self.dpvision_con6(x6)
        segout.append(deep_x6)
        fr_x6 = self.fr_concat6(fr_x6, x66)
        fr_x6 = self.BFI11(fr_x6)

        x7 = self.up4(x6)
        x7_ = x7
        fr_x7 = self.basic_conv7(fr_x6)
        x7 = self.concat7(x7, fr_x7, x0)
        x7 = self.BFI_d3(x7)
        fr_x7 = self.fr_concat7(fr_x7, x7)
        fr_x7 = self.BFI11(fr_x7)
        x7 = self.final_concat(x7, fr_x7)

        output = self.out_conv(x7)

        return output


    def name(self):
        return "BDPNet"

class BFI(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(BFI, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.InstanceNorm2d(features),
                nn.LeakyReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        # fea_v_1, fea_v_2 = fea_v.split(fea_v.shape[-1] // 2, dim=1)
        return fea_v

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels):
        """
: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)
        )

    def forward(self, x):
        x = self.conv_conv(x)
        return x

class MSFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        self.norm = nn.InstanceNorm2d(out_channels, **norm_op_kwargs)
        self.relu = nn.LeakyReLU(**net_nonlin_kwargs)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.norm(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        x3 = self.norm(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x)
        x4 = self.norm(x4)
        x4 = self.relu(x4)
        out = x1 + x2 + x3 + x4

        return out
class Basic1(nn.Module):
    """ Basic conv Block"""

    def __init__(self, in_channels, out_channels):
        super(Basic1, self).__init__()
        self.conv_hr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)
        )

    def forward(self, x):
        x = self.conv_hr(x)
        return x


class Basic(nn.Module):
    """ Basic conv Block"""

    def __init__(self, in_channels):
        super(Basic, self).__init__()
        self.conv_hr = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)
        )

    def forward(self, x):
        x = self.conv_hr(x)
        return x


class DownBlock(nn.Module):
    """Downsampling before ConvBlock"""

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            TPAP(stride=1, distortionmode=True),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownBlock_Con(nn.Module):
    """Downsampling """

    def __init__(self, pooling_p):
        super(DownBlock_Con, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pooling_p),
            # nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock_Con(nn.Module):
    """Upampling """

    def __init__(self, scale_factor=2):
        super(UpBlock_Con, self).__init__()
        self.uppool_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            # nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.uppool_conv(x)


class Concatenate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concatenate, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SpatialSELayer3D_0(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*

    """

    def __init__(self, in_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D_0, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, input_tensor_concat, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, H, W = input_tensor_concat.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(
            input_tensor_concat, squeeze_tensor.view(batch_size, 1, H, W))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*

    """

    def __init__(self, in_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, input_tensor_concat, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, H, W = input_tensor_concat.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(
            input_tensor_concat, squeeze_tensor.view(batch_size, 1, H, W))

        return output_tensor


class Concatenate_Threechs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concatenate_Threechs, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upssampling before ConvBlock"""

    def __init__(self, in_channels, out_channels,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.uppool_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvBlock(in_channels, out_channels)
            )

    def forward(self, x):
        x = self.uppool_conv(x)
        return x


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        新增modulation 参数： 是DCNv2中引入的调制标量
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 输出通道是2N
        nn.init.constant_(self.p_conv.weight, 0)  # 权重初始化为0
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:  # 如果需要进行调制
            # 输出通道是N
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)  # 在指定网络层执行完backward（）之后调用钩子函数

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):  # x: (b,c,h,w)
        offset = self.p_conv(x)  # (b,2N,h,w) 学习到的偏移量 2N表示在x轴方向的偏移和在y轴方向的偏移
        if self.modulation:  # 如果需要调制
            m = torch.sigmoid(self.m_conv(x))  # (b,N,h,w) 学习到的N个调制标量

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 如果需要调制
        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # (b,c,h,w,N)
            x_offset *= m  # 为偏移添加调制标量

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class TPAP(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(TPAP, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.wave_pool = DWT_2D(wavename='haar')
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:  # 是否调制
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lra)  # 在指定网络层执行完backward()之后调用钩子函数

            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lra(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    def forward(self, x):

        wav_out = self.wave_pool(x)
        wav_out = torch.mean(wav_out[0], dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out = self.downmax(max_out)
        out = torch.cat([max_out, wav_out], dim=1)
        # 如果需要调制
        if self.distortionmode:
            d_avg_out = torch.sigmoid(self.d_conv(wav_out))  # (b,N,h,w) 学习到的N个调制标量,试试out换成x
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out1 = d_avg_out * max_out
            out2 = d_max_out * wav_out
            out = torch.cat([out1, out2], dim=1)

        out = self.conv(out)
        mask = self.sigmoid(out)
        x = self.down(x)
        att_out = x * mask
        return F.relu(att_out)


if __name__ == "__main__":
    params = {'in_chns': 3,
              'class_num': 2,
              'feature_chns': [16, 32, 64, 128],
              'fr_feature_chns': 16,
              'bilinear': True,
              '_deep_supervision': True,
              'do_ds': True,
              'con_op': True}
    Net = BDPNet(params)
    print(Net)
    #     Net = Net.double()
    Net = Net.cuda()

    x = np.random.rand(2, 3, 40, 40, 40)
    xt = torch.from_numpy(x).float()
    xt = torch.tensor(xt).cuda()

    y = Net(xt)
    #     y = y.cpu()
    print('11', y[0].shape, 'done')
    # print(Net)
