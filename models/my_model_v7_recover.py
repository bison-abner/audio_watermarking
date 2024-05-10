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
        self.hinet = Hinet(num_layers=num_layers)
        self.watermark_fc = torch.nn.Linear(num_bit, num_point)
        self.watermark_fc_back = torch.nn.Linear(num_point, num_bit)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dropout1 = torch.nn.Dropout()
        self.identity = torch.nn.Identity()
        self.recover_layer = SameSizeConv2d(2, 2)
        self.use_recover_layer = use_recover_layer

    # def stft(self, data):
    #     window = torch.hann_window(self.n_fft).to(data.device)
    #     tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
    #     # [1, 501, 41, 2]
    #     return tmp

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        tmp = torch.view_as_real(tmp)
        # [1, 501, 41, 2]
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)

        # Changed in version 2.0: Real datatype inputs are no longer supported. Input must now have a complex datatype, as returned by stft(..., return_complex=True).

        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def encode(self, signal, message, need_fft=False):
        # 1.信号执行fft
        signal_fft = self.stft(signal)
        # import pdb
        # pdb.set_trace()
        # (batch,freq_bins,time_frames,2)

        # 2.Message执行fft
        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)

        # 3.encode
        signal_wmd_fft, msg_remain = self.enc_dec(signal_fft, message_fft, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd = self.istft(signal_wmd_fft)
        if need_fft:
            return signal_wmd, signal_fft, message_fft

        return signal_wmd

    def decode(self, signal):
        signal_fft = self.stft(signal)
        if self.use_recover_layer:
            signal_fft = self.recover_layer(signal_fft)
        watermark_fft = signal_fft
        # watermark_fft = torch.randn(signal_fft.shape).cuda()
        _, message_restored_fft = self.enc_dec(signal_fft, watermark_fft, rev=True)
        message_restored_expanded = self.istft(message_restored_fft)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        return message_restored_float

    def enc_dec(self, signal, watermark, rev):
        signal = signal.permute(0, 3, 2, 1)
        # [4, 2, 41, 501]

        watermark = watermark.permute(0, 3, 2, 1)

        # pdb.set_trace()
        signal2, watermark2 = self.hinet(signal, watermark, rev)
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)


class SameSizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SameSizeConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # (batch,501,41,2]
        x1 = x.permute(0, 3, 1, 2)
        # (batch,2,501,41]
        x2 = self.conv(x1)
        # (batch,2,501,41]
        x3 = x2.permute(0, 2, 3, 1)
        # (batch,501,41,2]
        return x3
