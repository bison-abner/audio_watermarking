import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform(audio_path, label, color):
    y, sr = librosa.load(audio_path, sr=16000)
    plt.plot(librosa.times_like(y, sr=sr), y, label=label, color=color)

# 创建一个图形
plt.figure(figsize=(10, 4))

# 绘制原始音频的波形图
plot_waveform(r'C:\Users\abner\Desktop\毕业设计\test_audio\LibriSpeech_1756-134819-0061.wav', 'Original Audio', 'blue')

# 绘制嵌入水印后音频的波形图
plot_waveform(r'C:\Users\abner\Desktop\学习\阿里云盘下载\43c5fec74f0ee06bcc3bbda5a2df88ffb1eb7be04a32344b18bc90a8.wav', 'Watermarked Audio', 'orange')

# 添加标题和标签
plt.title('Waveform Comparison')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
