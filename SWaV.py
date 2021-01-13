from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn

import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SwAV
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.optim.optimizer import Optimizer
from typing import Callable, Optional
from pytorch_lightning.utilities import AMPType


def cli_main():
    from pl_bolts.datamodules import STL10DataModule
    from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = SwAV.add_model_specific_args(parser)
    args = parser.parse_args()

    args.batch_size = 1024
    args.data_dir = "/scratch/sgt287/data"
    args.dataset = "stl10"
    args.arch = "resnet18"
    args.maxpool1 = False

    dm = STL10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    dm.train_dataloader = dm.train_dataloader_mixed
    dm.val_dataloader = dm.val_dataloader_mixed
    args.num_samples = dm.num_unlabeled_samples

    normalization = stl10_normalization()

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )


    model = SwAV(**args.__dict__)

    checkpoint_callback = ModelCheckpoint(monitor=None, period=10, save_top_k=-1, verbose=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=1,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=args.gpus > 1,
        precision=16,
        callbacks=[checkpoint_callback],
        fast_dev_run=args.fast_dev_run,
        auto_scale_batch_size=testing,
        default_root_dir="log/SWaV"
    )


    trainer.fit(model, dm)



if __name__ == '__main__':
    cli_main()
