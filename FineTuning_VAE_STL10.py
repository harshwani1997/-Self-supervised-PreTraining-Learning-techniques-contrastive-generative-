import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from VAE import VAE
#from pl_bolts.models.self_supervised.swav.transforms import SwAVFinetuneTransform
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEFineTuner(pl.LightningModule):

    def __init__(
        self,
        backbone = None,
        in_features = 512,
        num_classes = 10,
        hidden_dim = 128,
        learning_rate = 0.01,
        weight_decay = 1e-6
    ):

        super().__init__()

        self.hparams.lr = learning_rate
        self.weight_decay = weight_decay

        self.backbone = backbone
        self.decoder = self.block_forward = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes, bias=True)
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)


    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            features = self.backbone(x)
        features = features.view(len(features), -1)
        logits = self.decoder(features)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.val_acc(logits, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.test_acc(logits, y)

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3,
            patience = 5,
            mode = 'min'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }



def cli_main():
    from pl_bolts.datamodules import STL10DataModule

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--backbone_checkpoint', type=int, default=9)
    parser.add_argument('--data_dir', type=str, help='path to dataset', default='.')

    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--in_features', type=int, default=512)
    parser.add_argument('--hidden_features', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    args = parser.parse_args()

    dm = STL10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    dm.train_dataloader = dm.train_dataloader_labeled
    dm.val_dataloader = dm.val_dataloader_labeled
    args.num_samples = 0

    # dm.train_transforms = SwAVFinetuneTransform(
    #     normalize=stl10_normalization(),
    #     input_height=dm.size()[-1],
    #     eval_transform=False
    # )
    # dm.val_transforms = SwAVFinetuneTransform(
    #     normalize=stl10_normalization(),
    #     input_height=dm.size()[-1],
    #     eval_transform=True
    # )
    # dm.test_transforms = SwAVFinetuneTransform(
    #     normalize=stl10_normalization(),
    #     input_height=dm.size()[-1],
    #     eval_transform=True
    # )

    args.maxpool1 = False
    args.first_conv = True

    backbone_path = f'beta_VAE/lightning_logs/version_0/checkpoints/epoch={args.backbone_checkpoint}.ckpt'
    print(f"Backbone = {backbone_path}")
    backbone = VAE.load_from_checkpoint(backbone_path, strict=False).encoder

    model = VAEFineTuner(backbone,
        in_features=args.in_features,
        num_classes=dm.num_classes,
        hidden_dim=args.hidden_features,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        max_epochs=120,
        min_epochs=120,
        auto_lr_find=True,
        #callbacks=[early_stopping_callback],
        weights_save_path=f"./weights/FT/{args.backbone_checkpoint}/",
        default_root_dir=f"./log/FT/{args.backbone_checkpoint}/",
        progress_bar_refresh_rate=1000
    )
    trainer.tune(model, dm)

    print(dm)
    print(backbone)
    print(f"Backbone = {backbone_path}")
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    print(f"Backbone = {backbone_path}")


if __name__ == '__main__':
    cli_main()