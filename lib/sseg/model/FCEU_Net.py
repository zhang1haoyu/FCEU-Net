# -*- coding:utf-8  -*-
import torch
from torch import nn, einsum
from collections import OrderedDict
from lib.sseg.model.SSA import shunted_s, shunted_b
from torchsummary import summary
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from thop import profile
from torch.nn import init
import torch.nn.functional as F
from efficientnet_pytorch.model import MemoryEfficientSwish
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import Tuple, Optional
from einops import rearrange
from torch import Tensor, LongTensor
from torch.nn import Softmax
import math
import torch.nn as nn
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ptflops import get_model_complexity_info
import re


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='nearest')


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, U):
        q = self.conv1x1(U)  # 将特征图通过 1x1 卷积减少通道数
        q = self.activation(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, U):
        identity = U
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.relu(z)
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.activation(z)
        out = U * z.expand_as(U)
        return out  #jia identity


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)  # 将通道按照分组数目进行重新排列
        x = x.permute(0, 2, 1, 3, 4)  # 调整通道的排列顺序
        x = x.reshape(b, -1, h, w)  # 将通道重新展平为原始形状
        return x

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        combined = torch.cat([U_sse, U_cse], dim=1)
        gate = self.gate(combined)
        out = U_sse * gate + U_cse * (1 - gate)
        out = self.channel_shuffle(out,2)
        return out


class HiLo(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.3):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim

        # Lo-Fi的自注意力头数
        self.l_heads = int(num_heads * alpha)
        # Lo-Fi的token维度
        self.l_dim = self.l_heads * head_dim

        # Hi-Fi的自注意力头数
        self.h_heads = num_heads - self.l_heads
        # Hi-Fi的token维度
        self.h_dim = self.h_heads * head_dim

        # 本地窗口大小
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 等价于标准的多头自注意力
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # 低频注意力 (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size, padding=0, groups=dim, bias=False)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # 高频注意力 (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        # 将输入从 (B, C, H, W) 转换为 (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        if self.h_heads == 0:
            x = self.lofi(x)
            return x.permute(0, 3, 1, 2)  # 转换回 (B, C, H, W)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.permute(0, 3, 1, 2)  # 转换回 (B, C, H, W)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)

        # 转换回 (B, C, H, W)
        return x.permute(0, 3, 1, 2)

    def flops(self, H, W):
        # 当高度和宽度不能被窗口大小整除时，对特征图进行填充
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # 对于 Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # q @ k 和 attn @ v
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # 投影
        hifi_flops += Np * self.h_dim * self.h_dim

        # 对于 Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.l_dim
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        # k, v
        lofi_flops += kv_len * self.dim * self.l_dim * 2
        # q @ k 和 attn @ v
        lofi_flops += Np * self.l_dim * kv_len * 2
        # 投影
        lofi_flops += Np * self.l_dim * self.l_dim

        return hifi_flops + lofi_flops


class LWF(nn.Module):
    def __init__(self, dims=128, eps=1e-8):
        super(LWF, self).__init__()
        self.eps = eps

        # 使用非线性组合方式
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.nonlinear_combine = nn.PReLU()

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(dims),
            nn.Sigmoid()
        )

        # 融合后进行卷积、BN和ReLU
        self.post_conv = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU()
        )

    def forward(self, x, skip):
        # 计算融合权重并进行非线性组合
        weights = nn.ReLU()(self.weights)
        fuse_weights = self.nonlinear_combine(weights / (torch.sum(weights, dim=0) + self.eps))

        # 进行特征融合
        x = fuse_weights[0] * skip + fuse_weights[1] * x

        # 添加注意力机制
        attention_map = self.attention(x)
        x = x * attention_map

        # 卷积处理
        x = self.post_conv(x)

        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2,
                                          output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.deconv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x


class FECU_Net(nn.Module):
    def __init__(self, img_size=256, pretrained='', num_classes=1, vis=None, dropout =0.1):
        super(FECU_Net, self).__init__()
        self.encoder = shunted_s(img_size=img_size, pretrained=pretrained)

        self.WF4 = LWF(256)
        self.decoder4 = UpSampleBlock(in_channels=512, out_channels=256)

        self.WF3 = LWF(128)
        self.decoder3 = UpSampleBlock(in_channels=256, out_channels=128)

        self.WF2 = LWF(64)
        self.decoder2 = UpSampleBlock(in_channels=128, out_channels=64)

        self.WF1 = LWF(32)
        self.decoder1 = UpSampleBlock(in_channels=64, out_channels=32)
        #self.DEcoder = Decoder(encoder_channels = (64, 128, 256, 512), decode_channels=64,num_classes=32)
        self.HiLo1 = HiLo(64)
        self.HiLo2 = HiLo(128)
        self.HiLo3 = HiLo(256)
        self.HiLo4 = HiLo(512)
        self.attn4 = nn.ModuleList([scSE(256) for i in range(4)])
        self.attn3 = nn.ModuleList([scSE(128) for i in range(2)])
        self.attn2 = nn.ModuleList([scSE(64) for i in range(2)])
        self.scSE = scSE(512)
        self.down4 = nn.Sequential(
            nn.Conv2d(768, 256, 1, padding=0, bias=False),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(384, 128, 1, padding=0, bias=False),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(192, 64, 1, padding=0, bias=False),
        )

        self.down = nn.Sequential(
            nn.Conv2d(960, 512, 1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )



        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1, bias=False),
        )

    def forward(self, x):
        '''
        in:torch.Size([48, 3, 256, 256])
        e4:torch.Size([48, 512, 8, 8])     e3:torch.Size([48, 256, 16, 16])     e2:torch.Size([48, 128, 32, 32])    e1:torch.Size([48, 64, 64, 64])
        d4:torch.Size([48, 256, 16, 16])   d3:torch.Size([48, 128, 32, 32])     d2:torch.Size([48, 64, 64, 64])     d1:torch.Size([48, 32, 128, 128])
        out:torch.Size([48, 6, 256, 256])
        '''
        result = OrderedDict()
        e1, e2, e3, e4 = self.encoder(x)
        #e1 = self.simam1(e1)
        #e2 = self.simam1(e2)
        '''for blk in self.feat4:
            e4 = blk(e4)
        for blk in self.feat3:
            e3 = blk(e3)
        for blk in self.feat2:
            e2 = blk(e2)
        for blk in self.feat1:
            e1 = blk(e1)'''


        '''e4 = self.MHSA(e4)'''

        '''e4 = self.scSE(e4)'''

        e4 = self.HiLo4(e4)

        residual4 = Upsample(e4, 16)#512,16,16
        residual4_64 = Upsample(e4, 64)

        d4 = self.decoder4(e4)
        d4 = self.WF4(d4, e3)
        '''residual3 = Upsample(d4, 64)'''
        for blk in self.attn4:
            d4 = blk(d4)
        d4_plus = torch.cat([residual4, d4], dim=1) #768,16,16
        d4_plus = self.down4(d4_plus)#256,16,16
        residual3 = Upsample(d4_plus, 32)#256,32,32
        residual3_64 = Upsample(d4_plus, 64)

        d3 = self.decoder3(d4_plus)
        d3 = self.WF3(d3, e2)
        '''residual2 = Upsample(d3, 64)'''
        for blk in self.attn3:
            d3 = blk(d3)
        d3_plus = torch.cat([residual3, d3], dim=1)#384,32,32
        d3_plus = self.down3(d3_plus)#128,32,32
        residual2 = Upsample(d3_plus, 64)#128,64,64

        d2 = self.decoder2(d3_plus)
        d2 = self.WF2(d2, e1)
        '''residual1 = Upsample(d2, 64)'''
        for blk in self.attn2:
            d2 = blk(d2)
        d2_plus = torch.cat([residual2, d2], dim=1)#192,64,64
        d2_plus = self.down2(d2_plus)#64,64,64

        '''d_mix = torch.cat([d2_plus, residual2, residual3_64, residual4_64], dim=1)
        d_mix = self.down(d_mix)'''

        d1 = self.decoder1(d2_plus)
        #d1_1 = self.DEcoder(e1, e2, e3, e4, 128, 128)
        #d1 = self.WF1(d1_1, d1)


        out = self.final(d1)

        result['out'] = out
        return result


if __name__ == "__main__":

    tensor = torch.rand(1,3,256,256)
    model = HST_UNet_LWF(img_size=256)
    summary(model, (3, 256, 256), device='cpu')

'''    data = torch.randn((1, 3, 256, 256)).cuda()
    measure_inference_speed(model, (data,))'''


