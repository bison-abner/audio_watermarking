# src/evaluate.py
import numpy as np
import torch
import pypesq
import resampy


def signal_noise_ratio(original, signal_watermarked):
    original, signal_watermarked = to_equal_length(original, signal_watermarked)
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:  # 说明原始信号并未改变
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength
    ratio = max(1e-10, ratio)
    return 10 * np.log10(ratio)

def calc_ber(watermark_decoded_tensor, watermark_tensor, threshold=0.5):
    watermark_decoded_binary = watermark_decoded_tensor >= threshold
    watermark_binary = watermark_tensor >= threshold
    ber_tensor = 1 - (watermark_decoded_binary == watermark_binary).to(torch.float32).mean()
    return ber_tensor


def resample_to16k(data, sr):
    new_sr = 16000
    new_data = resampy.resample(data, sr, new_sr)
    return new_data


def pesq(signal1, signal2, sr):
    signal1, signal2 = to_equal_length(signal1, signal2)
    if sr != 16000:
        signal1 = resample_to16k(signal1, sr)
        signal2 = resample_to16k(signal2, sr)
    try:
        pesq_score = pypesq.pesq(signal1, signal2, 16000)
    except Exception as e:
        pesq_score = 0
        print(f"PESQ calculation error: {e}")
    return pesq_score

def to_equal_length(original, signal_watermarked):
    min_length = min(len(original), len(signal_watermarked))
    original = original[:min_length]
    signal_watermarked = signal_watermarked[:min_length]
    return original, signal_watermarked
