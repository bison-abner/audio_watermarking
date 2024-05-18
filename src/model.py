
import torch
import torch.nn as nn
from models.hinet import Hinet

class Model(nn.Module):
    def __init__(self, num_point, num_bit, n_fft, hop_length, use_recover_layer, num_layers):
        super(Model, self).__init__()
        self.hinet = Hinet(num_layers=num_layers)
        self.watermark_fc = nn.Linear(num_bit, num_point)
        self.watermark_fc_back = nn.Linear(num_point, num_bit)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dropout1 = nn.Dropout()
        self.identity = nn.Identity()
        self.recover_layer = SameSizeConv2d(2, 2)
        self.use_recover_layer = use_recover_layer

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        tmp = torch.view_as_real(tmp)
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)

    def encode(self, signal, message, need_fft=False):
        signal_fft = self.stft(signal)
        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)
        signal_wmd_fft, _ = self.enc_dec(signal_fft, message_fft, rev=False)
        signal_wmd = self.istft(signal_wmd_fft)
        if need_fft:
            return signal_wmd, signal_fft, message_fft
        return signal_wmd

    def decode(self, signal):
        signal_fft = self.stft(signal)
        if self.use_recover_layer:
            signal_fft = self.recover_layer(signal_fft)
        watermark_fft = signal_fft
        _ , message_restored_fft = self.enc_dec(signal_fft, watermark_fft, rev=True)
        message_restored_expanded = self.istft(message_restored_fft)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        return message_restored_float

    def enc_dec(self, signal, watermark, rev):
        signal = signal.permute(0, 3, 2, 1)
        watermark = watermark.permute(0, 3, 2, 1)
        signal2, watermark2 = self.hinet(signal, watermark, rev)
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)

class SameSizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SameSizeConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = x.permute(0, 3, 1, 2)
        x2 = self.conv(x1)
        x3 = x2.permute(0, 2, 3, 1)
        return x3
