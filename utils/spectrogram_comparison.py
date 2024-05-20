import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(audio_path, title, output_path=None):
    y, sr = librosa.load(audio_path, sr=16000)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

# 生成原始音频的频谱图
plot_spectrogram(r'C:\Users\abner\Desktop\毕业设计\test_audio\LibriSpeech_1756-134819-0061.wav', 'Original Audio Spectrogram', 'original_spectrogram.png')

# 生成嵌入水印后音频的频谱图
plot_spectrogram(r'C:\Users\abner\Desktop\学习\阿里云盘下载\43c5fec74f0ee06bcc3bbda5a2df88ffb1eb7be04a32344b18bc90a8.wav', 'Watermarked Audio Spectrogram', 'watermarked_spectrogram.png')
