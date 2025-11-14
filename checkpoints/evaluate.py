from tqdm.auto import tqdm
import hydra
import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn.modules import module
import torch.nn.functional as F
from torch.distributions import Categorical
from src import utils
from einops import rearrange, repeat, reduce

from train import SequenceLightningModule
from omegaconf import OmegaConf

import os

@hydra.main(config_path="../configs", config_name="generate.yaml")
def main(config: OmegaConf):
    # Load train config from existing Hydra experiment
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
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path smartly
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

        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(state_dict)
        model.eval()

    ## Test single batch
    debug = False
    if debug:
        val_dataloaders = model.val_dataloader()
        loader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

        model = model.to('cuda')
        model.eval()
        batch = next(iter(loader))
        batch = (batch[0].cuda(), batch[1].cuda(), batch[2])
        model.model.layers[0].layer.kernel()
        with torch.no_grad():
            x, y, *w = model.forward(batch)
            loss = model.loss_val(x, y, *w)
            print("Single batch loss:", loss)


    # Custom: Evaluate on training set if requested
    if getattr(config, 'eval_on_train', False):
        print("Evaluating on the training set...")
        train_loader = model.train_dataloader()
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in train_loader:
                x, y, *z = batch
                x = x.cuda()
                y = y.cuda()
                out, *_ = model.model(x)
                preds = out.argmax(-1)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        acc = (all_preds == all_targets).float().mean().item()
        print(f"Train set accuracy: {acc:.4f}")
    else:
        ## Use PL test to calculate final metrics
        from train import create_trainer
        trainer = create_trainer(config)
        trainer.test(model)

if __name__ == '__main__':
    main()
