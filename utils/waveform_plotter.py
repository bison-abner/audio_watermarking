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
plot_waveform(r'C:\Users\abner\Desktop\毕业设计\test_audio\LibriSpeech_1756-134819-0061.wav', '原始音频', 'original_waveform.png')

# 生成嵌入水印后音频的波形图
plot_waveform(r'C:\Users\abner\Desktop\学习\阿里云盘下载\43c5fec74f0ee06bcc3bbda5a2df88ffb1eb7be04a32344b18bc90a8.wav', '嵌入水印的音频', 'watermarked_waveform.png')
