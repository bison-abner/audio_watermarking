import os
import soundfile
import librosa
import resampy


def is_wav_file(filename):
    # 获取文件扩展名
    file_extension = os.path.splitext(filename)[1]

    # 判断文件扩展名是否为'.wav'或'.WAV'
    return file_extension.lower() == ".wav"


import numpy as np


def read_as_single_channel_16k(audio_file, def_sr, verbose=False, aim_second=None):
    assert os.path.exists(audio_file), "音频文件不存在"

    file_extension = os.path.splitext(audio_file)[1].lower()

    if file_extension == ".mp3":
        data, origin_sr = librosa.load(audio_file, sr=None)
    elif file_extension in [".wav", ".flac"]:
        data, origin_sr = soundfile.read(audio_file)
    else:
        raise Exception("不支持的文件类型:" + file_extension)

    # 通道数
    if len(data.shape) == 2:
        left_channel = data[:, 0]
        if verbose:
            print("双通道文件,变为单通道")
        data = left_channel

    # 采样率
    if origin_sr != def_sr:
        data = resampy.resample(data, origin_sr, def_sr)
        if verbose:
            print("原始音频采样率不是16kHZ,可能会对水印性能造成影响")

    sr = def_sr
    audio_length_second = 1.0 * len(data) / sr
    if verbose:
        print("输入音频长度:%d秒" % audio_length_second)

    # 判断通道数
    if len(data.shape) == 2:
        data = data[:, 0]
        print("选取第一个通道")

    if aim_second is not None:
        signal = data
        assert len(signal) > 0
        current_second = len(signal) / sr
        if current_second < aim_second:
            repeat_count = int(aim_second / current_second) + 1
            signal = np.repeat(signal, repeat_count)
        data = signal[0:sr * aim_second]

    return data, sr, audio_length_second


def read_as_single_channel(file, aim_sr):
    if file.endswith(".mp3"):
        data, sr = librosa.load(file, sr=aim_sr)  # 这里默认就是会转换为输入的sr
    else:
        data, sr = soundfile.read(file)

    if len(data.shape) == 2:  # 双声道
        data = data[:, 0]  # 只要第一个声道

    # 然后再切换sr,因为soundfile可能读取出一个双通道的东西
    if sr != aim_sr:
        data = resampy.resample(data, sr, aim_sr)
    return data