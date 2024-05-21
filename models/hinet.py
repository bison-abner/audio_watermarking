import torch
from models.invblock import INV_block


class Hinet(torch.nn.Module):
    # 初始化 Hinet 类
    def __init__(self, in_channel=2, num_layers=16):
        super(Hinet, self).__init__()
        self.inv_blocks = torch.nn.ModuleList([INV_block(in_channel) for _ in range(num_layers)])

    # 创建一个包含 16 个 INV_block 的模块列表，每个 INV_block 的输入通道数为 2
    def forward(self, x1, x2, rev=False):
        """
        前向传播和反向传播的函数
        参数:
        - x1: 封面数据 (cover)
        - x2: 秘密数据 (secret)
        - rev: 是否进行反向传播 (默认为 False)

        返回:
        - x1: 处理后的封面数据
        - x2: 处理后的秘密数据
        """
        # x1:cover
        # x2:secret
        if not rev:
            # 前向传播过程：依次通过每个可逆块进行处理
            for inv_block in self.inv_blocks:
                x1, x2 = inv_block(x1, x2)
        else:
            # 反向传播过程：依次通过每个可逆块的反向处理
            for inv_block in reversed(self.inv_blocks):
                x1, x2 = inv_block(x1, x2, rev=True)
        return x1, x2
