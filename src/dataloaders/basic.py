"""Implementation of basic benchmark datasets used in S4 experiments: MNIST, CIFAR10 and Speech Commands."""

import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange
from src.utils import permutations
import math

from src.dataloaders.base import default_data_path, ImageResolutionSequenceDataset, ResolutionSequenceDataset, SequenceDataset


class MNIST(SequenceDataset):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    l_output = 0
    L = 784

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        ]  # (L, d_input)
        if self.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        self.dataset_train = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.MNIST(
            self.data_dir,
            train=False,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CIFAR10(ImageResolutionSequenceDataset):
    _name_ = "cifar"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "permute": None,
            "grayscale": False,
            "tokenize": False,  # if grayscale, tokenize into discrete byte inputs
            "augment": False,
            "cutout": False,
            "rescale": None,
            "random_erasing": False,
            "val_split": 0.1,
            "seed": 42,  # For validation split
        }

    @property
    def d_input(self):
        if self.grayscale:
            if self.tokenize:
                return 256
            else:
                return 1
        else:
            assert not self.tokenize
            return 3

    def setup(self):
        img_size = 32
        if self.rescale:
            img_size //= self.rescale

        if self.grayscale:
            preprocessors = [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    lambda x: x.view(1, img_size * img_size).t()
                )  # (L, d_input)
            ]

            if self.tokenize:
                preprocessors.append(
                    torchvision.transforms.Lambda(lambda x: (x * 255).long())
                )
                permutations_list.append(Rearrange("l 1 -> l"))
            else:
                preprocessors.append(
                    torchvision.transforms.Normalize(
                        mean=122.6 / 255.0, std=61.0 / 255.0
                    )
                )
        else:
            preprocessors = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                ),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    Rearrange("z h w -> (h w) z", z=3, h=img_size, w=img_size)
                )  # (L, d_input)
            ]

        # Permutations and reshaping
        if self.permute == "br":
            permutation = permutations.bitreversal_permutation(img_size * img_size)
            print("bit reversal", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "snake":
            permutation = permutations.snake_permutation(img_size, img_size)
            print("snake", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "hilbert":
            permutation = permutations.hilbert_permutation(img_size)
            print("hilbert", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "transpose":
            permutation = permutations.transpose_permutation(img_size, img_size)
            transform = torchvision.transforms.Lambda(
                lambda x: torch.cat([x, x[permutation]], dim=-1)
            )
            permutations_list.append(transform)
        elif self.permute == "2d":  # h, w, c
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> h w c", h=img_size, w=img_size)
                )
            permutations_list.append(permutation)
        elif self.permute == "2d_transpose":  # c, h, w
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> c h w", h=img_size, w=img_size)
                )
            permutations_list.append(permutation)

        # Augmentation
        if self.augment:
            augmentations = [
                torchvision.transforms.RandomCrop(
                    img_size, padding=4, padding_mode="symmetric"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
            ]

            post_augmentations = []
            if self.cutout:
                post_augmentations.append(Cutout(1, img_size // 2))
                pass
            if self.random_erasing:
                # augmentations.append(RandomErasing())
                pass
        else:
            augmentations, post_augmentations = [], []
        transforms_train = (
            augmentations + preprocessors + post_augmentations + permutations_list
        )
        transforms_eval = preprocessors + permutations_list

        transform_train = torchvision.transforms.Compose(transforms_train)
        transform_eval = torchvision.transforms.Compose(transforms_eval)
        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}",
            train=True,
            download=True,
            transform=transform_train,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}", train=False, transform=transform_eval
        )

        if self.rescale:
            print(f"Resizing all images to {img_size} x {img_size}.")
            self.dataset_train.data = self.dataset_train.data.reshape((self.dataset_train.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)).max(4).max(2).astype(np.uint8)
            self.dataset_test.data = self.dataset_test.data.reshape((self.dataset_test.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)).max(4).max(2).astype(np.uint8)

        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"

class SpeechCommands(ResolutionSequenceDataset):
    _name_ = "sc"

    @property
    def init_defaults(self):
        return {
            "mfcc": False,
            "dropped_rate": 0.0,
            "length": 16000,
            "all_classes": False,
        }

    @property
    def d_input(self):
        _d_input = 20 if self.mfcc else 1
        _d_input += 1 if self.dropped_rate > 0.0 else 0
        return _d_input

    @property
    def d_output(self):
        return 10 if not self.all_classes else 35

    @property
    def l_output(self):
        return 0

    @property
    def L(self):
        return 161 if self.mfcc else self.length


    def setup(self):
        self.data_dir = self.data_dir or default_data_path # TODO make same logic as other classes

        from src.dataloaders.datasets.sc import _SpeechCommands

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommands(
            partition="train",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_val = _SpeechCommands(
            partition="val",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_test = _SpeechCommands(
            partition="test",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

class FSC(ResolutionSequenceDataset):
    _name_ = "fsc"

    @property
    def init_defaults(self):
        return {
            "mfcc": True,
            "n_mfcc": 20,
            "n_mels": 32,
            "n_fft": 400,
            "hop_length": 160,
            "melkwargs": None,
            "dropped_rate": 0.0,
            "length": 16000,
            "sample_rate": 16000,
            "all_classes": False,
        }

    @property
    def d_input(self):
        # Return appropriate input dimension based on mfcc flag
        if self.mfcc:
            return self.n_mfcc  # Number of MFCC coefficients
        else:
            return 1  # Raw waveform is 1D

    @property
    def d_output(self):
        # Return actual number of action classes from dataset
        return self._d_output

    @property
    def l_output(self):
        return 0  # Classification task

    @property
    def L(self):
        # Return sequence length based on mfcc flag
        if self.mfcc:
            # Calculate number of frames based on STFT parameters
            return math.ceil(self.length / self.hop_length)
        else:
            return self.length

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "fluent_speech_commands_dataset"
        
        # Whether to use full FSC or just the action labels
        if self.all_classes:
            from src.dataloaders.datasets.fsc_full import get_fsc_datasets
        else:
            from src.dataloaders.datasets.fsc import get_fsc_datasets
            
        # from src.dataloaders.datasets.fsc import get_fsc_datasets
        
        # from src.dataloaders.datasets.fsc_full import get_fsc_datasets

        # Build melkwargs (only used if mfcc is True)
        if self.melkwargs is not None:
            melkwargs = self.melkwargs
        else:
            melkwargs = {
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.n_fft,
                "center": True,
                "f_min": 0.0,
            }

        train, val, test = get_fsc_datasets(
            self.data_dir,
            max_length=self.length,
            target_sr=self.sample_rate,
            mfcc=self.mfcc,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            melkwargs=melkwargs,
            dropped_rate=self.dropped_rate,  # Pass dropped_rate to dataset
        )

        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

        # Derive dynamic dimensions
        if self.mfcc:
            # MFCC mode: (frames, n_mfcc)
            sample, _ = train[0]
            self._d_input = sample.shape[1]  # n_mfcc
            self._L = sample.shape[0]  # number of frames
        else:
            # Raw waveform mode: (length, 1)
            sample, _ = train[0]
            self._d_input = sample.shape[1]  # 1
            self._L = sample.shape[0]

        self._d_output = len(train.intents)  # Number of action classes (should be 6)


# New: FSC with random split from training set (80/10/10)
class FSC_RandomSplit(ResolutionSequenceDataset):
    _name_ = "fsc_randomsplit"

    @property
    def init_defaults(self):
        return {
            "mfcc": True,
            "n_mfcc": 20,
            "n_mels": 32,
            "n_fft": 400,
            "hop_length": 160,
            "melkwargs": None,
            "dropped_rate": 0.0,
            "length": 16000,
            "sample_rate": 16000,
        }

    @property
    def d_input(self):
        if self.mfcc:
            return self.n_mfcc
        else:
            return 1

    @property
    def d_output(self):
        return self._d_output

    @property
    def l_output(self):
        return 0

    @property
    def L(self):
        if self.mfcc:
            return math.ceil(self.length / self.hop_length)
        else:
            return self.length

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "fluent_speech_commands_dataset"
        from src.dataloaders.datasets.fsc_randomsplit import get_fsc_datasets

        if self.melkwargs is not None:
            melkwargs = self.melkwargs
        else:
            melkwargs = {
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.n_fft,
                "center": True,
                "f_min": 0.0,
            }

        train, val, test = get_fsc_datasets(
            self.data_dir,
            max_length=self.length,
            target_sr=self.sample_rate,
            mfcc=self.mfcc,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            melkwargs=melkwargs,
            # dropped_rate=self.dropped_rate,
        )

        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

        if self.mfcc:
            sample, _ = train[0]
            self._d_input = sample.shape[1]
            self._L = sample.shape[0]
        else:
            sample, _ = train[0]
            self._d_input = sample.shape[1]
            self._L = sample.shape[0]

        self._d_output = len(train.intents)


class FSCImage(SequenceDataset):
    """FSC dataset that outputs 2D spectrograms as images for vision models."""
    
    _name_ = "fsc_image"

    @property
    def init_defaults(self):
        return {
            "n_mfcc": 40,
            "n_mels": 80,
            "n_fft": 400,
            "hop_length": 160,
            "dropped_rate": 0.0,
            "max_length": 16000,
            "target_sr": 16000,
        }

    @property
    def d_input(self):
        # For vision models, this is typically channels (1 for grayscale spectrogram)
        return 1

    @property
    def d_output(self):
        return self._d_output

    @property
    def l_output(self):
        return 0  # Classification task

    @property
    def L(self):
        # For 2D images, L is not really used, but we return total pixels
        # Image shape: (1, n_mfcc, n_frames)
        n_frames = math.ceil(self.max_length / self.hop_length)
        return self.n_mfcc * n_frames

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "fluent_speech_commands_dataset"
        
        from src.dataloaders.datasets.fsc_image import get_fsc_image_datasets

        train, val, test = get_fsc_image_datasets(
            self.data_dir,
            max_length=self.max_length,
            target_sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            dropped_rate=self.dropped_rate,
        )

        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

        # Get sample to determine dimensions
        # Output shape: (1, n_mfcc, n_frames)
        sample, _ = train[0]
        self._d_output = len(train.intents)  # Number of action classes (should be 6)
        
        print(f"FSC Image dataset loaded: {len(train)} train, {len(val)} val, {len(test)} test")
        print(f"Image shape: {sample.shape}, Output classes: {self._d_output}")

class FSCAug(ResolutionSequenceDataset):
    _name_ = "fsc_aug"

    @property
    def init_defaults(self):
        return {
            "mfcc": True,
            "n_mfcc": 40,
            "n_mels": 64,
            "n_fft": 400,
            "hop_length": 160,
            "melkwargs": None,
            "dropped_rate": 0.0,
            "length": 16000,
            "sample_rate": 16000,
            "augment": True,
            "time_mask_param": 30,
            "freq_mask_param": 15,
        }

    @property
    def d_input(self):
        return self.n_mfcc if self.mfcc else 1

    @property
    def d_output(self):
        return self._d_output

    @property
    def l_output(self):
        return 0

    @property
    def L(self):
        if self.mfcc:
            return math.ceil(self.length / self.hop_length)
        else:
            return self.length

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "fluent_speech_commands_dataset"
        from src.dataloaders.datasets.fsc_aug import get_fsc_aug_datasets

        if self.melkwargs is not None:
            melkwargs = self.melkwargs
        else:
            melkwargs = {
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.n_fft,
                "center": True,
                "f_min": 0.0,
            }

        train, val, test = get_fsc_aug_datasets(
            self.data_dir,
            max_length=self.length,
            target_sr=self.sample_rate,
            mfcc=self.mfcc,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            melkwargs=melkwargs,
            dropped_rate=self.dropped_rate,
            augment=self.augment,
            time_mask_param=self.time_mask_param,
            freq_mask_param=self.freq_mask_param,
        )

        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

        if self.mfcc:
            sample, _ = train[0]
            self._d_input = sample.shape[1]
            self._L = sample.shape[0]
        else:
            sample, _ = train[0]
            self._d_input = sample.shape[1]
            self._L = sample.shape[0]

        self._d_output = len(train.intents)


class FSCMultiLabel(ResolutionSequenceDataset):
    _name_ = "fsc_multilabel"

    @property
    def init_defaults(self):
        return {
            "mfcc": True,
            "n_mfcc": 64,
            "n_mels": 80,
            "n_fft": 400,
            "hop_length": 160,
            "melkwargs": None,
            "dropped_rate": 0.0,
            "length": 16000,
            "sample_rate": 16000,
            "all_classes": True,
        }

    @property
    def d_input(self):
        return self.n_mfcc if self.mfcc else 1

    @property
    def d_output(self):
        return self._d_output

    @property
    def l_output(self):
        return 0

    @property
    def L(self):
        if self.mfcc:
            return math.ceil(self.length / self.hop_length)
        else:
            return self.length

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "fluent_speech_commands_dataset"
        from src.dataloaders.datasets.fsc_multilabel import FSCMultiLabelDataset

        if self.melkwargs is not None:
            melkwargs = self.melkwargs
        else:
            melkwargs = {
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.n_fft,
                "center": True,
                "f_min": 0.0,
            }

        # Create train dataset first to get label mappings
        train = FSCMultiLabelDataset(
            self.data_dir,
            split="train",
            max_length=self.length,
            target_sr=self.sample_rate,
            mfcc=self.mfcc,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            melkwargs=melkwargs,
            dropped_rate=self.dropped_rate,
        )

        # Use train mappings for val and test
        val = FSCMultiLabelDataset(
            self.data_dir,
            split="valid",
            max_length=self.length,
            target_sr=self.sample_rate,
            mfcc=self.mfcc,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            melkwargs=melkwargs,
            dropped_rate=self.dropped_rate,
            component_label2idx=train.label2idx,
        )

        test = FSCMultiLabelDataset(
            self.data_dir,
            split="test",
            max_length=self.length,
            target_sr=self.sample_rate,
            mfcc=self.mfcc,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            melkwargs=melkwargs,
            dropped_rate=self.dropped_rate,
            component_label2idx=train.label2idx,
        )

        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

        sample, labels = train[0]
        if self.mfcc:
            self._d_input = sample.shape[1]
            self._L = sample.shape[0]
        else:
            self._d_input = sample.shape[1]
            self._L = sample.shape[0]

        # For multilabel, d_output should be tuple of (n_actions, n_objects, n_locations)
        self._d_output = (
            len(train.label2idx['action']),
            len(train.label2idx['object']),
            len(train.label2idx['location'])
        )
