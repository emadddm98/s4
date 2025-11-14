"""Fluent Speech Commands dataset loader for intent classification."""

import os
import pathlib
import math
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class FSCIntentDataset(Dataset):
    def __init__(
        self,
        root,
        split: str = "train",
        transform=None,
        max_length: int = 16000,              # Raw waveform target length (in samples)
        target_sr: int | None = None,
        resample_lowpass_width: int = 64,
        apply_resample: bool = True,
        mfcc: bool = False,
        n_mfcc: int = 40,
        n_mels: int = 64,
        melkwargs: dict | None = None,
        dropped_rate: float = 0.0,            # Fraction of dataset to DROP (0.0-1.0)
        # New: precomputed mapping from action label -> index (must come from train split)
        intent2idx: dict | None = None,
    ):
        self.root = pathlib.Path(root)
        self.split = split
        self.transform = transform
        self.max_length = int(max_length)

        # Resample controls
        self.target_sr = target_sr
        self.resample_lowpass_width = resample_lowpass_width
        self.apply_resample = apply_resample

        # MFCC / mel parameters
        self.mfcc = bool(mfcc)
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)

        # Build melkwargs with sensible defaults (torchaudio MFCC -> MelSpectrogram)
        self.melkwargs = melkwargs.copy() if melkwargs else {}
        self.melkwargs.setdefault("n_mels", self.n_mels)
        # Common speech defaults
        self.melkwargs.setdefault("n_fft", 400)         # 25 ms @16k
        self.melkwargs.setdefault("hop_length", 160)    # 10 ms hop
        self.melkwargs.setdefault("win_length", 400)
        self.melkwargs.setdefault("center", True)
        self.melkwargs.setdefault("f_min", 0.0)

        self.n_fft = self.melkwargs["n_fft"]
        self.hop_length = self.melkwargs["hop_length"]
        self.center = self.melkwargs.get("center", True)

        # Cache of MFCC transforms per (sample_rate) to avoid re-instantiation
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
            # keep_count = max(1, int(len(self.meta) * keep_rate))
            # Use stratified sampling to maintain label distribution
            self.meta = self.meta.groupby('action', group_keys=False).apply(
                lambda x: x.sample(frac=keep_rate, random_state=56789)
            ).reset_index(drop=True)
            print(f"FSC {split}: Subsampled to {len(self.meta)} examples (dropped_rate={dropped_rate:.2f})")

        # Build or reuse global label mapping (actions only). We ONLY build it on the train split.
        if intent2idx is None:
            if split != "train":
                raise ValueError(
                    "intent2idx mapping must be provided for split!='train'. Build it from the training split first."
                )
            # Build mapping deterministically from training metadata
            self.intents = sorted(self.meta["action"].unique())
            self.intent2idx = {intent: idx for idx, intent in enumerate(self.intents)}
        else:
            self.intent2idx = intent2idx
            # Derive ordered list of intents (sorted by index to preserve original ordering)
            self.intents = [intent for intent, _ in sorted(self.intent2idx.items(), key=lambda kv: kv[1])]

        # Sanity check: ensure all actions in this split exist in mapping
        missing = set(self.meta["action"].unique()) - set(self.intent2idx.keys())
        if missing:
            raise ValueError(
                f"Found action labels in split '{split}' not present in training mapping: {missing}"
            )

        # Pre-compute (fixed) feature length when using MFCC so downstream
        # sequence models see consistent L.
        if self.mfcc:
            # torchaudio MFCC (which wraps MelSpectrogram) with center=True pads
            # reflectively by n_fft//2 on both sides, so #frames ≈ ceil(max_length / hop_length)
            if self.center:
                self.feature_max_length = math.ceil(self.max_length / self.hop_length)
            else:
                # Classic STFT framing formula
                self.feature_max_length = max(
                    1, 1 + (self.max_length - self.n_fft) // self.hop_length
                )
        else:
            self.feature_max_length = None  # Not used in raw waveform mode

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
        waveform, sr = torchaudio.load(str(wav_path))  # (C, T)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform  # (1,T)
        # Resample if requested
        if self.apply_resample and self.target_sr is not None and sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr,
                lowpass_filter_width=self.resample_lowpass_width,
            )
            waveform = resampler(waveform)
            sr = self.target_sr
        return waveform, sr  # (1, T'), sr

    def _pad_truncate_1d(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        L = x.shape[-1]
        if L < target_len:
            x = torch.nn.functional.pad(x, (0, target_len - L))
        elif L > target_len:
            x = x[..., :target_len]
        return x

    def _pad_truncate_frames(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        # x: (frames, feat)
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

        waveform, sr = self._load_audio(wav_path)  # (1, T)

        # Always enforce a fixed raw length before feature extraction (for speed & consistency)
        waveform = self._pad_truncate_1d(waveform, self.max_length)  # (1, max_length)

        # # DEBUG: Print on first sample
        # if idx == 0:
        #     print(f"\n{'='*60}")
        #     print(f"[DATASET DEBUG] Split: {self.split}")
        #     print(f"[DATASET DEBUG] Using MFCC: {self.mfcc}")
        #     print(f"[DATASET DEBUG] Sample rate: {sr}")
        #     print(f"[DATASET DEBUG] Waveform shape after pad/truncate: {waveform.shape}")
        #     print(f"[DATASET DEBUG] Max length setting: {self.max_length}")
        #     print(f"{'='*60}\n")

        if self.mfcc:
            # MFCC expects (batch/channel, time)
            mfcc_transform = self._get_mfcc_transform(sr)
            # torchaudio MFCC output: (channel, n_mfcc, frames)
            mfcc = mfcc_transform(waveform)  # (1, n_mfcc, F)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (F, n_mfcc)
            # Pad/truncate frames dimension to fixed feature_max_length
            mfcc = self._pad_truncate_frames(mfcc, self.feature_max_length)
            mfcc = mfcc.float()
            if self.transform:
                mfcc = self.transform(mfcc)
            label = self.intent2idx[row["action"]]
            # if idx == 0:
            #     print(f"[DATASET DEBUG] Output MFCC shape: {mfcc.shape}")
            return mfcc, label  # Shape: (Frames, n_mfcc)
        else:
            # Raw waveform path -> (T,) then (T,1)
            wav_1d = waveform.squeeze(0).float()  # (max_length,)
            if self.transform:
                wav_1d = self.transform(wav_1d)
            wav_2d = wav_1d.unsqueeze(-1)  # (L, 1)
            label = self.intent2idx[row["action"]]
            # if idx == 0:
            #     print(f"[DATASET DEBUG] Output raw waveform shape: {wav_2d.shape}")
            return wav_2d, label

def get_fsc_datasets(
    root,
    max_length: int = 16000,
    target_sr: int | None = None,
    melkwargs: dict | None = None,
    **kwargs,
):
    """Create train/val/test datasets with a CONSISTENT action label mapping.

    The mapping is built from the training split and re-used for valid/test to
    guarantee consistent indices across splits.
    """
    train = FSCIntentDataset(
        root,
        split="train",
        max_length=max_length,
        target_sr=target_sr,
        melkwargs=melkwargs,
        **kwargs,
    )
    mapping = train.intent2idx
    val = FSCIntentDataset(
        root,
        split="valid",
        max_length=max_length,
        target_sr=target_sr,
        melkwargs=melkwargs,
        intent2idx=mapping,
        **kwargs,
    )
    test = FSCIntentDataset(
        root,
        split="test",
        max_length=max_length,
        target_sr=target_sr,
        melkwargs=melkwargs,
        intent2idx=mapping,
        **kwargs,
    )
    return train, val, test