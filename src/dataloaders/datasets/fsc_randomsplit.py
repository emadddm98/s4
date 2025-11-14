"""Fluent Speech Commands dataset loader for intent classification with random split from training data."""

import os
import pathlib
import math
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class FSCIntentDatasetSplit(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        audio_dir: pathlib.Path,
        transform=None,
        max_length: int = 16000,
        target_sr: int | None = None,
        resample_lowpass_width: int = 64,
        apply_resample: bool = True,
        mfcc: bool = False,
        n_mfcc: int = 40,
        n_mels: int = 64,
        melkwargs: dict | None = None,
        intent2idx: dict | None = None,
    ):
        self.meta = meta.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transform = transform
        self.max_length = int(max_length)
        self.target_sr = target_sr
        self.resample_lowpass_width = resample_lowpass_width
        self.apply_resample = apply_resample
        self.mfcc = bool(mfcc)
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.melkwargs = melkwargs.copy() if melkwargs else {}
        self.melkwargs.setdefault("n_mels", self.n_mels)
        self.melkwargs.setdefault("n_fft", 400)
        self.melkwargs.setdefault("hop_length", 160)
        self.melkwargs.setdefault("win_length", 400)
        self.melkwargs.setdefault("center", True)
        self.melkwargs.setdefault("f_min", 0.0)
        self.n_fft = self.melkwargs["n_fft"]
        self.hop_length = self.melkwargs["hop_length"]
        self.center = self.melkwargs.get("center", True)
        self._mfcc_transforms: dict[int, torchaudio.transforms.MFCC] = {}
        if intent2idx is None:
            self.intents = sorted(self.meta["action"].unique())
            self.intent2idx = {intent: idx for idx, intent in enumerate(self.intents)}
        else:
            self.intent2idx = intent2idx
            self.intents = [intent for intent, _ in sorted(self.intent2idx.items(), key=lambda kv: kv[1])]
        missing = set(self.meta["action"].unique()) - set(self.intent2idx.keys())
        if missing:
            raise ValueError(f"Found action labels not present in mapping: {missing}")
        if self.mfcc:
            if self.center:
                self.feature_max_length = math.ceil(self.max_length / self.hop_length)
            else:
                self.feature_max_length = max(1, 1 + (self.max_length - self.n_fft) // self.hop_length)
        else:
            self.feature_max_length = None

    def __len__(self):
        return len(self.meta)

    def _get_mfcc_transform(self, sample_rate: int):
        if sample_rate not in self._mfcc_transforms:
            self._mfcc_transforms[sample_rate] = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs=self.melkwargs,
            )
        return self._mfcc_transforms[sample_rate]

    def _load_audio(self, wav_path: pathlib.Path):
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.apply_resample and self.target_sr is not None and sr != self.target_sr:
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

    def _pad_truncate_frames(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        F = x.shape[0]
        if F < target_frames:
            pad = torch.zeros(target_frames - F, x.shape[1], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        elif F > target_frames:
            x = x[:target_frames]
        return x

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        wav_path = self.audio_dir / row["path"]
        waveform, sr = self._load_audio(wav_path)
        waveform = self._pad_truncate_1d(waveform, self.max_length)
        if self.mfcc:
            mfcc_transform = self._get_mfcc_transform(sr)
            mfcc = mfcc_transform(waveform)
            mfcc = mfcc.squeeze(0).transpose(0, 1)
            mfcc = self._pad_truncate_frames(mfcc, self.feature_max_length)
            mfcc = mfcc.float()
            if self.transform:
                mfcc = self.transform(mfcc)
            label = self.intent2idx[row["action"]]
            return mfcc, label
        else:
            wav_1d = waveform.squeeze(0).float()
            if self.transform:
                wav_1d = self.transform(wav_1d)
            wav_2d = wav_1d.unsqueeze(-1)
            label = self.intent2idx[row["action"]]
            return wav_2d, label

def get_fsc_datasets(
    root,
    max_length: int = 16000,
    target_sr: int | None = None,
    melkwargs: dict | None = None,
    random_state: int = 42,
    **kwargs,
):
    """Create train/val/test datasets by splitting the original training set (80/10/10)."""
    root = pathlib.Path(root)
    train_csv = root / "data" / "train_data.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Training metadata file {train_csv} not found.")
    meta = pd.read_csv(train_csv)
    # Build mapping from training split
    intents = sorted(meta["action"].unique())
    intent2idx = {intent: idx for idx, intent in enumerate(intents)}
    # Stratified split: 80% train, 20% temp
    meta_train, meta_temp = train_test_split(
        meta, test_size=0.2, stratify=meta["action"], random_state=random_state
    )
    # Split temp into 10% val, 10% test (relative to original)
    meta_val, meta_test = train_test_split(
        meta_temp, test_size=0.5, stratify=meta_temp["action"], random_state=random_state
    )
    audio_dir = root
    train_set = FSCIntentDatasetSplit(
        meta_train, audio_dir, max_length=max_length, target_sr=target_sr, melkwargs=melkwargs, intent2idx=intent2idx, **kwargs
    )
    val_set = FSCIntentDatasetSplit(
        meta_val, audio_dir, max_length=max_length, target_sr=target_sr, melkwargs=melkwargs, intent2idx=intent2idx, **kwargs
    )
    test_set = FSCIntentDatasetSplit(
        meta_test, audio_dir, max_length=max_length, target_sr=target_sr, melkwargs=melkwargs, intent2idx=intent2idx, **kwargs
    )
    return train_set, val_set, test_set
