import hydra
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from train import SequenceLightningModule
from src import utils
import os

@hydra.main(config_path="../configs", config_name="generate.yaml")
def main(config: OmegaConf):
    # Load train config from existing Hydra experiment if provided
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    if config.train.seed is not None:
        import pytorch_lightning as pl
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path
    if not config.experiment_path:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    else:
        ckpt_path = os.path.join(config.experiment_path, config.checkpoint_path)
    print("Full checkpoint path:", ckpt_path)

    # Load model
    if ckpt_path.endswith('.ckpt'):
        model = SequenceLightningModule.load_from_checkpoint(ckpt_path, config=config)
        model.to('cuda')
    elif ckpt_path.endswith('.pt'):
        model = SequenceLightningModule(config)
        model.to('cuda')
        state_dict = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(state_dict)
        model.eval()

    # User must provide an audio file path
    audio_path = getattr(config, 'audio_path', None)
    if audio_path is None:
        raise ValueError("Please provide an audio_path=<path/to/audio.wav> as a Hydra override.")
    audio_path = hydra.utils.to_absolute_path(audio_path)
    print(f"Loading audio from: {audio_path}")

    # Preprocess audio (match your dataset's pipeline)
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)  # Convert to mono
    # Resample if needed (assume 16kHz target)
    target_sr = getattr(config.dataset, 'sample_rate', 16000)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    # Pad/truncate to match model's expected input length
    max_length = getattr(config.dataset, 'length', 16000)
    if wav.shape[1] < max_length:
        wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]))
    elif wav.shape[1] > max_length:
        wav = wav[:, :max_length]
    # (L,) -> (L, 1)
    wav = wav.squeeze(0).unsqueeze(-1)
    # Add batch dimension
    wav = wav.unsqueeze(0).to('cuda')  # (1, L, 1)

    # Run model
    model.eval()
    with torch.no_grad():
        logits, *_ = model.model(wav)
        pred = logits.argmax(-1).item()
        print(f"Predicted class: {pred}")

if __name__ == '__main__':
    main()
