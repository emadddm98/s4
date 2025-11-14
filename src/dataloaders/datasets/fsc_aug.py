"""Fluent Speech Commands dataset loader with SpecAugment for intent classification."""

import os
import pathlib
import math
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import torchaudio.transforms as T

class FSCIntentAugDataset(Dataset):
    def __init__(
        self,
        root,
        split: str = "train",
        max_length: int = 16000,
        target_sr: int | None = None,
        resample_lowpass_width: int = 64,
        apply_resample: bool = True,
        mfcc: bool = False,
        n_mfcc: int = 40,
        n_mels: int = 64,
        melkwargs: dict | None = None,
        dropped_rate: float = 0.0,
        augment: bool = True,
        time_mask_param: int = 30,
        freq_mask_param: int = 15,
    ):
        self.root = pathlib.Path(root)
        self.split = split
        self.max_length = int(max_length)
        self.target_sr = target_sr
        self.resample_lowpass_width = resample_lowpass_width
        self.apply_resample = apply_resample
        self.mfcc = bool(mfcc)
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.augment = augment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
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
        meta_file = self.root / "data" / f"{split}_data.csv"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file {meta_file} not found.")
        self.meta = pd.read_csv(meta_file)
        self.audio_dir = self.root
        if dropped_rate > 0.0:
            keep_rate = 1.0 - dropped_rate
            self.meta = self.meta.groupby('action', group_keys=False).apply(
                lambda x: x.sample(frac=keep_rate, random_state=56789)
            ).reset_index(drop=True)
            print(f"FSC {split}: Subsampled to {len(self.meta)} examples (dropped_rate={dropped_rate:.2f})")
        self.intents = sorted(self.meta["action"].unique())
        self.intent2idx = {intent: idx for idx, intent in enumerate(self.intents)}
        self.d_output = len(self.intents)
        if self.mfcc:
            print("Using MFCC features")
            if self.center:
                self.feature_max_length = math.ceil(self.max_length / self.hop_length)
            else:
                self.feature_max_length = max(
                    1, 1 + (self.max_length - self.n_fft) // self.hop_length
                )
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
    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        x = T.TimeMasking(time_mask_param=self.time_mask_param)(x)
        x = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(x)
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
            if self.augment and self.split == "train":
                mfcc = self._spec_augment(mfcc)
            label = self.intent2idx[row["action"]]
            return mfcc, label
        else:
            wav_1d = waveform.squeeze(0).float()
            wav_2d = wav_1d.unsqueeze(-1)
            label = self.intent2idx[row["action"]]
            return wav_2d, label

def get_fsc_aug_datasets(
    root,
    max_length: int = 16000,
    target_sr: int | None = None,
    melkwargs: dict | None = None,
    **kwargs,
):
    train = FSCIntentAugDataset(
        root,
        split="train",
        max_length=max_length,
        target_sr=target_sr,
        melkwargs=melkwargs,
        **kwargs,
    )
    val_kwargs = dict(kwargs)
    val_kwargs["augment"] = False
    val = FSCIntentAugDataset(
        root,
        split="valid",
        max_length=max_length,
        target_sr=target_sr,
        melkwargs=melkwargs,
        **val_kwargs,
    )
    test_kwargs = dict(kwargs)
    test_kwargs["augment"] = False
    test = FSCIntentAugDataset(
        root,
        split="test",
        max_length=max_length,
        target_sr=target_sr,
        melkwargs=melkwargs,
        **test_kwargs,
    )
    return train, val, test
