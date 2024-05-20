import numpy as np
import librosa


def calculate_snr(original_audio, watermarked_audio):
    # 计算信号和噪声
    signal = original_audio
    noise = original_audio - watermarked_audio

    # 计算信号功率和噪声功率
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    # 计算SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# 加载音频文件
original_audio_path = r'C:\Users\abner\Desktop\毕业设计\test_audio\LibriSpeech_1756-134819-0061.wav'
watermarked_audio_path = r'C:\Users\abner\Desktop\学习\阿里云盘下载\43c5fec74f0ee06bcc3bbda5a2df88ffb1eb7be04a32344b18bc90a8.wav'

original_audio, sr = librosa.load(original_audio_path, sr=None)
watermarked_audio, sr = librosa.load(watermarked_audio_path, sr=None)

# 确保两个音频文件的长度一致
min_len = min(len(original_audio), len(watermarked_audio))
original_audio = original_audio[:min_len]
watermarked_audio = watermarked_audio[:min_len]

# 计算并打印SNR
snr_value = calculate_snr(original_audio, watermarked_audio)
print(f'Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB')
