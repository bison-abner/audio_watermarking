import pdb

import torch.optim
import torch.nn as nn
from models.hinet import Hinet
# from utils.attacks import attack_layer, mp3_attack_v2, butterworth_attack
import numpy as np
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, num_point, num_bit, n_fft, hop_length, use_recover_layer, num_layers):
        super(Model, self).__init__()
        self.hinet = Hinet(num_layers=num_layers)  # 实例化 Hinet 模块
        self.watermark_fc = torch.nn.Linear(num_bit, num_point)  # 初始化水印全连接层
        self.watermark_fc_back = torch.nn.Linear(num_point, num_bit)  # 初始化反向水印全连接层
        self.n_fft = n_fft  # 设置 FFT 窗口大小
        self.hop_length = hop_length  # 设置 FFT 步长
        self.dropout1 = torch.nn.Dropout()  # 定义 dropout 操作
        self.identity = torch.nn.Identity()  # 定义 identity 操作
        self.recover_layer = SameSizeConv2d(2, 2)  # 初始化恢复层
        self.use_recover_layer = use_recover_layer  # 是否使用恢复层的标志

    # def stft(self, data):
    #     window = torch.hann_window(self.n_fft).to(data.device)
    #     tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
    #     # [1, 501, 41, 2]
    #     return tmp

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)  # 创建汉宁窗口
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)  # 执行短时傅里叶变换
        tmp = torch.view_as_real(tmp)  # 将复数表示转换为实数表示
        # [1, 501, 41, 2]
        return tmp  # 返回变换结果

    def istft(self, signal_wmd_fft):  # 创建汉宁窗口
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)  # 执行反向短时傅里叶变换

        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def encode(self, signal, message, need_fft=False):
        # 1.信号执行fft
        signal_fft = self.stft(signal) # 对信号执行短时傅里叶变换

        # 2.Message执行fft
        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)  # 对水印执行短时傅里叶变换

        # 3.encode
        signal_wmd_fft, msg_remain = self.enc_dec(signal_fft, message_fft, rev=False)  # 执行编码操作
        signal_wmd = self.istft(signal_wmd_fft)  # 对编码后的信号执行反向短时傅里叶变换
        if need_fft:
            return signal_wmd, signal_fft, message_fft  # 返回编码后的信号、信号的频域表示以及水印的频域表示

        return signal_wmd  # 返回编码后的信号

    def decode(self, signal):
        signal_fft = self.stft(signal)  # 对信号执行短时傅里叶变换
        if self.use_recover_layer:
            signal_fft = self.recover_layer(signal_fft)  # 如果使用恢复层，则对信号的频域表示执行恢复层操作
        watermark_fft = signal_fft  # 将信号的频域表示作为水印的频域表示
        # watermark_fft = torch.randn(signal_fft.shape).cuda()
        _, message_restored_fft = self.enc_dec(signal_fft, watermark_fft, rev=True)
        message_restored_expanded = self.istft(message_restored_fft)  # 对解码后的水印执行反向短时傅里叶变换
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)  # 对恢复后的水印执行反向全连接层操作，并进行截断操作
        return message_restored_float  # 返回恢复后的水印

    def enc_dec(self, signal, watermark, rev):
        signal = signal.permute(0, 3, 2, 1)  # 调整信号的维度顺序
        # [4, 2, 41, 501]

        watermark = watermark.permute(0, 3, 2, 1)  # 调整水印的维度顺序

        # pdb.set_trace()
        signal2, watermark2 = self.hinet(signal, watermark, rev)  # 将信号和水印输入 Hinet 模块执行编码/解码操作
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)  # 返回编码/解码后的信号和水印


class SameSizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SameSizeConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 定义 1x1 卷积层

    def forward(self, x):
        # (batch,501,41,2]
        x1 = x.permute(0, 3, 1, 2)  # 调整输入的维度顺序
        # (batch,2,501,41]
        x2 = self.conv(x1)  # 执行卷积操作
        # (batch,2,501,41]
        x3 = x2.permute(0, 2, 3, 1)  # 调整输出的维度顺序
        # (batch,501,41,2]
        return x3  # 返回卷积结果
