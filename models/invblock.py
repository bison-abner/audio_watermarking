import torch
import torch.nn as nn
from models.rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    def __init__(self, channel=2, subnet_constructor=ResidualDenseBlock_out, clamp=2.0):
        super().__init__()
        self.clamp = clamp

        # 初始化三个子网络，分别用于计算ρ、η和φ
        # ρ网络用于计算掩模参数
        # 控制水印嵌入时的变换程度，即对原始数据的扭曲程度。
        # 掩模参数的作用是影响嵌入水印的强度和位置，从而在不损失数据质量的情况下，在原始数据中嵌入水印信息。
        self.r = subnet_constructor(channel, channel)
        # η网络用于计算平移参数
        # 用于平移数据，即在原始数据上加上一个平移量。
        # 平移参数的作用是调整嵌入水印时数据的位置，以便更好地与水印信息进行融合。
        self.y = subnet_constructor(channel, channel)
        # φ网络用于计算可逆激活函数参数
        # 定义可逆激活函数的参数，用于对数据进行可逆的非线性变换。
        # 可逆激活函数的作用是在嵌入水印时引入一种非线性变换，以增加数据的复杂性和水印的鲁棒性。
        self.f = subnet_constructor(channel, channel)

    def e(self, s):
        # 定义可逆激活函数
        # 返回值：可逆激活函数的输出
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, rev=False):
        # 正向传播函数
        # 输入参数：
        #     x1: 输入数据1
        #     x2: 输入数据2
        #     rev: 是否执行逆向传播（默认为False）
        if not rev:
            # 正向传播过程

            # 计算第一个子网络的输出
            t2 = self.f(x2)
            # 计算x1和第一个子网络输出的和
            y1 = x1 + t2
            # 计算第二个子网络的输出
            s1, t1 = self.r(y1), self.y(y1)
            # 计算x2的变换后的输出
            y2 = self.e(s1) * x2 + t1

        else:
            # 逆向传播过程

            # 计算第一个子网络的输出

            s1, t1 = self.r(x1), self.y(x1)
            # 计算x2的变换后的输出
            y2 = (x2 - t1) / self.e(s1)
            # 计算第二个子网络的输出
            t2 = self.f(y2)
            # 计算x1的变换后的输出
            y1 = (x1 - t2)

        return y1, y2