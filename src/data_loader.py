
import os
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
import resampy

class AudioDataset(Dataset):
    def __init__(self, audio_files, watermarks, sr=16000):
        self.audio_files = audio_files
        self.watermarks = watermarks
        self.sr = sr

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, _ = sf.read(self.audio_files[idx])
        audio = resampy.resample(audio, orig_sr=_, target_sr=self.sr)
        watermark = self.watermarks[idx]
        return torch.FloatTensor(audio), torch.FloatTensor(watermark)

def load_data(audio_dir, watermark_file, batch_size=32, sr=16000):
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    watermarks = np.load(watermark_file)
    dataset = AudioDataset(audio_files, watermarks, sr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Usage:
# train_loader = load_data('path/to/train_audio', 'path/to/train_watermarks.npy')
# val_loader = load_data('path/to/val_audio', 'path/to/val_watermarks.npy')
