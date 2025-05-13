from torchsummary import summary
import math
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as  F
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

# 注意力机制模块
import torch.nn as nn
import torch
from einops import rearrange
import math

import torch
import torch.nn as nn
import math


def kernel_size(in_channel):  # 计算一维卷积的核大小，利用的是ECA注意力中的参数[动态卷积核]
    k = int((math.log2(in_channel) + 1) // 2)
    return k + 1 if k % 2 == 0 else k


class MultiScaleFeatureExtractor(nn.Module):  # 多尺度特征提取器[对T1和T2不同时刻的特征进入到不同尺寸的卷积核加强提取]

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 使用不同尺寸的卷积核进行特征提取
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 分别使用不同尺寸的卷积核进行卷积操作
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out = out1 + out2 + out3  # 将不同尺度的特征相加
        return out


class ChannelAttention(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        # 使用自适应平均池化和最大池化提取全局信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        # 使用一维卷积计算通道注意力
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        # 对 t1 和 t2 进行平均池化和最大池化
        t1_channel_avg_pool = self.avg_pool(t1)
        t1_channel_max_pool = self.max_pool(t1)
        t2_channel_avg_pool = self.avg_pool(t2)
        t2_channel_max_pool = self.max_pool(t2)
        # 将池化结果拼接并转换维度
        channel_pool = torch.cat([
            t1_channel_avg_pool, t1_channel_max_pool,
            t2_channel_avg_pool, t2_channel_max_pool
        ], dim=2).squeeze(-1).transpose(1, 2)
        # 使用一维卷积计算通道注意力
        t1_channel_attention = self.channel_conv1(channel_pool)
        t2_channel_attention = self.channel_conv2(channel_pool)
        # 堆叠并使用Softmax进行归一化
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention], dim=0)
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)
        return channel_stack


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用二维卷积计算空间注意力
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        # 计算 t1 和 t2 的平均值和最大值
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]
        # 将平均值和最大值拼接
        spatial_pool = torch.cat([
            t1_spatial_avg_pool, t1_spatial_max_pool,
            t2_spatial_avg_pool, t2_spatial_max_pool
        ], dim=1)
        # 使用二维卷积计算空间注意力
        t1_spatial_attention = self.spatial_conv1(spatial_pool)
        t2_spatial_attention = self.spatial_conv2(spatial_pool)
        # 堆叠并使用Softmax进行归一化
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)
        spatial_stack = self.softmax(spatial_stack)
        return spatial_stack


class TFAM(nn.Module):
    """时序融合注意力模块"""

    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, t1, t2):
        # 计算通道和空间注意力
        channel_stack = self.channel_attention(t1, t2)
        spatial_stack = self.spatial_attention(t1, t2)
        # 加权相加并进行融合
        stack_attention = channel_stack + spatial_stack + 1
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2
        return fuse


class BFM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.multi_scale_extractor = MultiScaleFeatureExtractor(in_channel, in_channel)
        self.tfam = TFAM(in_channel)

    def forward(self, t1, t2):
        # 进行多尺度特征提取
        t1_multi_scale = self.multi_scale_extractor(t1)
        t2_multi_scale = self.multi_scale_extractor(t2)
        # 使用TFAM进行融合
        output = self.tfam(t1_multi_scale, t2_multi_scale)
        return output


if __name__ == '__main__':
    model = BFM(in_channel=32).cuda()
    t1 = torch.randn(1, 32, 32, 32).cuda()
    t2 = torch.randn(1, 32, 32, 32).cuda()
    output = model(t1,
                   t2)  # 这里记住是两个输入,不要被名字唬住了。双时，说通俗点，就是两个不同时刻在同一时间的进行一个融合，或者是在同一个时刻将两个不同的输入进行融合。这里取个名字双时，就是两个输入进行融合，你也可以多加一个T3，叫三时，变成三个输入的融合，完全没问题。
    print(output.shape)

'''if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64).cuda()  # 随机生成输入数据
    model = MDFA(dim_in=32, dim_out=32).cuda()  # 实例化模块
    output = model(input)  # 将输入通过模块处理
    print(output.shape)  # 输出处理后的数据形状

    summary(model, (32, 64, 64))  # 使用 torchsummary 打印模型详细信息
'''




'''只加门级
AS4 = self.ASPP4(e4)   #32,512,8,8
        print('AS4', AS4.shape)
        seg_body_e4, seg_edge_e4 = self.squeeze_body_edge(AS4) #32,512,8,8
        print('seg_body', seg_body_e4.shape)
        dec0_fine = self.bot_fine(e3) #32,48,16,16
        print('dec0', dec0_fine.shape)
        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge_e4, 16), dec0_fine], dim=1))#32,512,16,16
        print('seg_edge', seg_edge.shape)
        seg_out = seg_edge + Upsample(seg_body_e4, 16) #32,512,16,16
        seg_out = self.con1(seg_out)#32,256,16,16
        print('seg_out', seg_out.shape)
'''
