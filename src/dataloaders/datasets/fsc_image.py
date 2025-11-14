"""Fluent Speech Commands dataset loader for image-based models (ViT).
Outputs 2D spectrograms as images for vision transformer models.
"""

import os
import pathlib
import math
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class FSCImageDataset(Dataset):
    """FSC dataset that outputs 2D spectrograms as images (C, H, W) for ViT models."""
    
    def __init__(
        self,
        root,
        split: str = "train",
        transform=None,
        max_length: int = 16000,
        target_sr: int | None = None,
        resample_lowpass_width: int = 64,
        apply_resample: bool = True,
        n_mfcc: int = 40,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        dropped_rate: float = 0.0,
    ):
        self.root = pathlib.Path(root)
        self.split = split
        self.transform = transform
        self.max_length = int(max_length)

        # Resample controls
        self.target_sr = target_sr if target_sr is not None else 16000
        self.resample_lowpass_width = resample_lowpass_width
        self.apply_resample = apply_resample

        # Spectrogram parameters
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

        # Create MFCC transform
        self._mfcc_transforms: dict[int, torchaudio.transforms.MFCC] = {}
        
        # Load metadata
        meta_file = self.root / "data" / f"{split}_data.csv"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file {meta_file} not found.")
        self.meta = pd.read_csv(meta_file)
        self.audio_dir = self.root

        # Subsample dataset if dropped_rate > 0
        if dropped_rate > 0.0:
            keep_rate = 1.0 - dropped_rate
            self.meta = self.meta.groupby('action', group_keys=False).apply(
                lambda x: x.sample(frac=keep_rate, random_state=56789)
            ).reset_index(drop=True)
            print(f"FSC {split}: Subsampled to {len(self.meta)} examples (dropped_rate={dropped_rate:.2f})")

        # Build label mapping
        self.intents = sorted(self.meta["action"].unique())
        self.intent2idx = {intent: idx for idx, intent in enumerate(self.intents)}

        # Calculate fixed spectrogram dimensions
        # With center=True, frames = ceil(max_length / hop_length)
        self.n_frames = math.ceil(self.max_length / self.hop_length)
        
        print(f"FSC Image Dataset ({split}): Output shape will be ({self.n_mfcc}, {self.n_frames}, 1) - (H, W, C) format for patch2d encoder")

    def __len__(self):
        return len(self.meta)

    def _get_mfcc_transform(self, sample_rate: int):
        if sample_rate not in self._mfcc_transforms:
            melkwargs = {
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.n_fft,
                "center": True,
                "f_min": 0.0,
            }
            self._mfcc_transforms[sample_rate] = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs=melkwargs,
            )
        return self._mfcc_transforms[sample_rate]

    def _load_audio(self, wav_path: pathlib.Path):
        waveform, sr = torchaudio.load(str(wav_path))
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if self.apply_resample and sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr,
                lowpass_filter_width=self.resample_lowpass_width,
            )
            waveform = resampler(waveform)
            sr = self.target_sr
        return waveform, sr

    def _pad_truncate_1d(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        L = x.shape[-1]
        if L < target_len:
            x = torch.nn.functional.pad(x, (0, target_len - L))
        elif L > target_len:
            x = x[..., :target_len]
        return x

    def _pad_truncate_2d(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Pad or truncate the time dimension (last dim) of spectrogram."""
        # x shape: (C, H, W) where W is time frames
        F = x.shape[-1]
        if F < target_frames:
            # Pad on the right
            pad = torch.zeros(*x.shape[:-1], target_frames - F, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif F > target_frames:
            x = x[..., :target_frames]
        return x

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        wav_path = self.audio_dir / row["path"]

        # Load and preprocess audio
        waveform, sr = self._load_audio(wav_path)  # (1, T)
        waveform = self._pad_truncate_1d(waveform, self.max_length)  # (1, max_length)

        # Extract MFCC as 2D spectrogram
        mfcc_transform = self._get_mfcc_transform(sr)
        # Output: (1, n_mfcc, frames)
        mfcc = mfcc_transform(waveform)  
        
        # Ensure fixed frame dimension
        mfcc = self._pad_truncate_2d(mfcc, self.n_frames)  # (1, n_mfcc, n_frames)
        
        # Apply any additional transforms
        if self.transform:
            mfcc = self.transform(mfcc)
        
        label = self.intent2idx[row["action"]]
        
        # Return as (H, W, C) for patch2d encoder compatibility
        # patch2d encoder expects (batch, H, W, C) format
        # So we permute from (C, H, W) to (H, W, C)
        mfcc = mfcc.permute(1, 2, 0)  # (n_mfcc, n_frames, 1)
        
        return mfcc.float(), label


def get_fsc_image_datasets(
    root,
    max_length: int = 16000,
    target_sr: int | None = None,
    **kwargs,
):
    """Get train/val/test datasets for image-based models."""
    train = FSCImageDataset(
        root,
        split="train",
        max_length=max_length,
        target_sr=target_sr,
        **kwargs,
    )
    val = FSCImageDataset(
        root,
        split="valid",
        max_length=max_length,
        target_sr=target_sr,
        **kwargs,
    )
    test = FSCImageDataset(
        root,
        split="test",
        max_length=max_length,
        target_sr=target_sr,
        **kwargs,
    )
    return train, val, test
