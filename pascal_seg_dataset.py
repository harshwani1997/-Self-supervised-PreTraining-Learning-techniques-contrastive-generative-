import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms

from pl_bolts.models.self_supervised.swav.transforms import SwAVFinetuneTransform
from pl_bolts.transforms.dataset_normalizations import stl10_normalization



class VOCSegmentationDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers = 8,
        img_size=(500, 300),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = 1234
        self.img_size = img_size
        self.batch_size = batch_size


    @property
    def num_classes(self):
        return 22


    def seg_transform(self, segmap):
        segmap = transforms.CenterCrop(self.img_size)(segmap)
        segmap = torch.from_numpy(np.array(segmap)).long()
        segmap[segmap == 255] = 21
        return segmap

    def img_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            stl10_normalization()
    ])

    def img_aug_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(self.img_size),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.ToTensor(),
            stl10_normalization()
    ])



    def prepare_data(self):
        VOCSegmentation(self.data_dir, image_set="train", download=True)
        VOCSegmentation(self.data_dir, image_set="val", download=True)

    def train_dataloader(self):
        dataset = VOCSegmentation(
            self.data_dir, image_set="train",
            transform=self.img_aug_transform(),
            target_transform=self.seg_transform
        )
        dataset, _ = random_split(dataset, [1200, 264],
                               generator=torch.Generator().manual_seed(self.seed))

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        dataset = VOCSegmentation(
            self.data_dir, image_set="train",
            transform=self.img_transform(),
            target_transform=self.seg_transform
        )
        _, dataset = random_split(dataset, [1200, 264],
                               generator=torch.Generator().manual_seed(self.seed))

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        dataset = VOCSegmentation(
            self.data_dir, image_set="val",
            transform=self.img_transform(),
            target_transform=self.seg_transform
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def size(self, dims=None):
        return (500, 281)




if __name__ == '__main__':
    dm = VOCSegmentationDataModule(data_dir="../data", batch_size=16)
    tl = dm.train_dataloader()

    from collections import Set
    s = set()
    for x, y in tl:
        for e in list(torch.unique(y)):
            s.add(e.item())
        print(s)
        breakpoint()