import pytorch_lightning as pl

from torch.utils.data import DataLoader
from . import image_dataset
from . import pretraining_dataset
from .. import builder


class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = pretraining_dataset.MultimodalPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=1000,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=1000,
            num_workers=self.cfg.train.num_workers,
        )


