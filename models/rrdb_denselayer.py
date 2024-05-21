import torch
import torch.nn as nn
import models.module_util as mutil


# 密集连接
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1, 1, bias=bias)  # 第一个卷积层
        self.conv2 = nn.Conv2d(in_channel + 32, 32, 3, 1, 1, bias=bias)  # 第二个卷积层
        self.conv3 = nn.Conv2d(in_channel + 2 * 32, 32, 3, 1, 1, bias=bias)  # 第三个卷积层
        self.conv4 = nn.Conv2d(in_channel + 3 * 32, 32, 3, 1, 1, bias=bias)  # 第四个卷积层
        self.conv5 = nn.Conv2d(in_channel + 4 * 32, out_channel, 3, 1, 1, bias=bias)  # 第五个卷积层
        self.lrelu = nn.LeakyReLU(inplace=True)  # LeakyReLU 激活函数
        # 初始化
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))  # 第一层卷积操作
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))  # 第二层卷积操作
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))  # 第三层卷积操作
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))  # 第四层卷积操作
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))  # 第五层卷积操作
        return x5  # 返回结果
