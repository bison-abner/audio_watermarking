import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    """
    初始化神经网络层的权重。

    Args:
        net_l (list or nn.Module): 包含神经网络模块或单个模块的列表。
        scale (float): 权重初始化的尺度因子。

    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming Normal 初始化卷积层权重。
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # 为残差块调整权重初始化尺度
                if m.bias is not None:
                    # 初始化偏置为零。
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # 使用 Kaiming Normal 初始化全连接层权重。
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    # 初始化偏置为零。
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 将批归一化层权重初始化为1，偏置初始化为0。
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    """
    构造包含多个相同类型层的序列。

    Args:
        block (nn.Module): 要重复的块的类型。
        n_layers (int): 要创建的层的数量。

    Returns:
        nn.Sequential: 重复层的顺序容器。
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''无批归一化的残差块
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        """
        初始化无批归一化的残差块。

        Args:
            nf (int): 输入通道数和输出通道数。

        """
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 初始化卷积层权重
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        """
        无批归一化的残差块的前向传播。

        Args:
            x (torch.Tensor): 输入张 量。

        Returns:
            torch.Tensor: 经过残差块处理后的输出张 量。

        """
        identity = x
        out = F.relu(self.conv1(x), inplace=True)  # 对第一个卷积层的输出应用 ReLU 激活函数
        out = self.conv2(out)  # 应用第二个卷积层
        return identity + out  # 将原始输入与第二个卷积层的输出相加


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """使用光流场对图像或特征图进行变换
    Args:
        x (Tensor): 尺寸为 (N, C, H, W) 的输入。
        flow (Tensor): 尺寸为 (N, H, W, 2) 的光流场，值在正常范围内。
        interp_mode (str): 'nearest' 或 'bilinear'
        padding_mode (str): 'zeros' 或 'border' 或 'reflection'
    Returns:
        Tensor: 变换后的图像或特征图
    """
    flow = flow.permute(0,2,3,1)
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # 网格
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # 宽(x)，高(y)，2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # 将网格缩放到 [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    # 使用 grid_sample 函数对图像或特征图进行采样
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output