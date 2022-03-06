from typing import Optional, Tuple
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from udons.datamodules.datasets.jigsawaudio_dataset import AudioFolderJigsawDataset
from omegaconf import OmegaConf


def siamese_collate(batch):
    """
    This collator is used in Jigsaw approach.
    Input:
        batch: Example
                batch = [
                    {"data": [img1,], "label": [lbl1, ]},        #img1
                    {"data": [img2,], "label": [lbl2, ]},        #img2
                    .
                    .
                    {"data": [imgN,], "label": [lblN, ]},        #imgN
                ]
                where:
                    img{x} is a tensor of size: num_towers x C x H x W
                    lbl{x} is an integer
    Returns: Example output:
                torch.tensor([img1_0, ..., imgN_0]),
                torch.tensor([lbl1, ..., lblN])

                where the output is of dimension: (N * num_towers) x C x H x W

    see https://github.com/facebookresearch/vissl/blob/aa3f7cc33b3b7806e15593083aedc383d85e4a53/vissl/data/collators/siamese_collator.py
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    num_data_sources = 1
    batch_size = len(batch)
    data = [x["data"] for x in batch]
    labels = [x["label"] for x in batch]

    # each image is of shape: num_towers x C x H x W
    # num_towers x C x H x W -> N x num_towers x C x H x W
    idx_data = torch.stack([data[i] for i in range(batch_size)])
    idx_labels = [labels[i] for i in range(batch_size)]
    batch_size, num_siamese_towers, channels, height, width = idx_data.size()
    # N x num_towers x C x H x W -> (N * num_towers) x C x H x W
    idx_data = idx_data.view(batch_size * num_siamese_towers, channels, height, width)
    should_flatten = False
    if idx_labels[0].ndim == 1:
        should_flatten = True
    idx_labels = torch.stack(idx_labels).squeeze()
    if should_flatten:
        idx_labels = idx_labels.flatten()

    return idx_data, idx_labels


class ConcatMinDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class ConcatMaxDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets: tuple):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
class JigsawAudioDataModule(LightningDataModule):
    """
    Example of LightningDataModule dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        paths: dict = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_rate: float = 44100.0,
        nb_timesteps: int = 44100 * 5,
        nb_channels: int = 1,
        patch_len: int = 32,
        nb_patches: int = 5,
        nb_classes: int = 120,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 27.5,
        f_max: int = 16000,
        n_mels: int = 256,
        patch_jitter_min: int = 5,
        patch_jitter_max: int = 12,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.paths = paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_rate = sample_rate

        self.nb_channels = nb_channels
        self.nb_timesteps = nb_timesteps
        self.patch_len = patch_len
        self.nb_patches = nb_patches
        self.nb_classes = nb_classes
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.patch_jitter_min = patch_jitter_min
        self.patch_jitter_max = patch_jitter_max

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # self.dims is returned when you call datamodule.size()
        self.dims = None
        resolver_name = "datamodule"
        OmegaConf.register_new_resolver(
            resolver_name,
            lambda name: getattr(self, name),
            use_cache=False
        )

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        common_args = {
            "nb_channels": self.nb_channels,
            "sample_rate": self.sample_rate,
            "patch_len": self.patch_len,
            "nb_patches": self.nb_patches,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "n_mels": self.n_mels,
            "patch_jitter_min": self.patch_jitter_min,
            "patch_jitter_max": self.patch_jitter_max,
        }

        datasets = {}
        for dset in ["train", "valid", "test"]:
            sets = []
            for path in self.paths:
                if path.get(f"{dset}_dir"):
                    sets.append(
                        AudioFolderJigsawDataset(
                            root=Path(path["root_dir"], path[f"{dset}_dir"]),
                            random_chunk_length=self.nb_timesteps,
                            oversample_factor=path["oversample_factor"],
                            **common_args
                        )
                    )
            datasets[dset] = torch.utils.data.ConcatDataset(sets)

        self.train_set = datasets["train"]
        self.valid_set = datasets["valid"]
        self.test_set = datasets["test"]

        self.dims = self.train_set[0]["data"][0].shape

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=siamese_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=siamese_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=siamese_collate,
        )
