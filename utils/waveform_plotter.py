import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform(audio_path, title, output_path=None):
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=16000)
    plt.figure(figsize=(10, 4))
    # 生成波形图
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    # 如果提供了输出路径，则保存波形图
    if output_path:
        plt.savefig(output_path)
    plt.show()

# 生成原始音频的波形图
plot_waveform('original_audio.wav', '原始音频', 'original_waveform.png')

# 生成嵌入水印后音频的波形图
plot_waveform('watermarked_audio.wav', '嵌入水印的音频', 'watermarked_waveform.png')
