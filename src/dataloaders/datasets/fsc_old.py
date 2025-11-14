"""Fluent Speech Commands dataset loader for intent classification."""

import os
import pathlib
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class FSCIntentDataset(Dataset):
    def __init__(self, root, split="train", transform=None, max_length=16000, target_sr=None, resample_lowpass_width=64, apply_resample=True):
        self.root = pathlib.Path(root)
        self.split = split
        self.transform = transform
        self.max_length = max_length

        # NEW PATCH
        # target_sr: desired sample rate (Hz). If None, keep original file sr.
        self.target_sr = target_sr
        # width parameter passed to torchaudio Resample lowpass filter
        self.resample_lowpass_width = resample_lowpass_width
        self.apply_resample = apply_resample

        # Load metadata
        meta_file = self.root / "data" / f"{split}_data.csv"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file {meta_file} not found.")

        self.meta = pd.read_csv(meta_file)
        # print("LENGTH OF META:", len(self.meta))
        self.audio_dir = self.root

        # Build intent label mapping
        self.intents = sorted(self.meta["action"].unique())
        ## EMAD: Print the intents for debugging
        # print(self.intents)

        self.intent2idx = {intent: idx for idx, intent in enumerate(self.intents)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        
        wav_path = self.audio_dir / row["path"]
        waveform, sr = torchaudio.load(str(wav_path))
        waveform = waveform.mean(dim=0)  # mono, if waveform.ndim > 1:

        # Resample with anti-aliasing if requested and target_sr differs from original
        if self.apply_resample and (self.target_sr is not None) and (self.target_sr != sr):
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr, lowpass_filter_width=self.resample_lowpass_width)
            waveform = resampler(waveform)  # (1, new_time)
            sr = self.target_sr

        waveform = waveform.squeeze(0)  # (time,)

        # Pad or truncate
        if len(waveform) < self.max_length:
            pad = self.max_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:self.max_length]

        label = self.intent2idx[row["action"]]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform.unsqueeze(1), label  # (L, 1), label

def get_fsc_datasets(root, max_length=16000, target_sr=None, **kwargs):
    train = FSCIntentDataset(root, split="train", max_length=max_length, target_sr=target_sr, **kwargs)
    val = FSCIntentDataset(root, split="valid", max_length=max_length, target_sr=target_sr, **kwargs)
    test = FSCIntentDataset(root, split="test", max_length=max_length, target_sr=target_sr, **kwargs)
    return train, val, test