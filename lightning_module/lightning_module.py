from pathlib import Path
from typing import Sequence

import torch
from models import VAE
from pytorch_lightning import LightningModule
from torch import Tensor
from torchvision.utils import save_image


class MyLightningModule(VAE, LightningModule):
    def __init__(
        self,
        *,
        batch_size: int,
        lr: float = 0.001,
        weight_decay: float = 0,
        **kwargs,
    ) -> None:
        args_names = set(
            super().__init__.__code__.co_varnames[
                1 : super().__init__.__code__.co_kwonlyargcount + 1
            ]
        )
        super().__init__(**{k: v for (k, v) in kwargs.items() if k in args_names})
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer], []

    def training_step(self, batch: Sequence[Tensor], batch_idx: int):
        image, labels = batch
        results = self.forward(image)
        loss = self.loss_function(**results)

        self.log('metrics/train_loss', loss['loss'].item(), prog_bar=True)
        self.log(
            'metrics/train_reconstruction_loss',
            loss['Reconstruction_Loss'].item(),
        )
        self.log('metrics/train_KLD', loss['KLD'].item())

        return loss['loss']

    def validation_step(self, batch: Sequence[Tensor], batch_idx: int):
        image, labels = batch
        results = self.forward(image)
        loss = self.loss_function(**results)

        self.log('metrics/val_loss', loss['loss'].item(), prog_bar=True)
        self.log('metrics/val_reconstruction_loss', loss['Reconstruction_Loss'].item())
        self.log('metrics/val_KLD', loss['KLD'].item())

        return {'image': image}

    def validation_epoch_end(self, outputs):
        self.log_images(
            torch.cat([x['image'] for x in outputs], 0)[: 12 ** 2],
        )

    def log_images(self, test_input: Tensor):
        recons = self.generate(test_input)

        save_dir = Path(self.logger.log_dir, 'images')
        save_dir.mkdir(exist_ok=True)

        save_image(
            recons.data,
            save_dir / f"recons_image-{self.current_epoch}.png",
            normalize=False,
            nrow=12,
            pad_value=0.5,
        )

        save_image(
            test_input.data,
            save_dir / f"real_image-{self.current_epoch}.png",
            normalize=False,
            nrow=12,
            pad_value=0.5,
        )

        error = recons - test_input
        error = torch.cat([error * (error > 0), error * 0, -error * (error < 0)], 1)
        save_image(
            error.data,
            save_dir / f"error_image-{self.current_epoch}.png",
            normalize=False,
            nrow=12,
            pad_value=0.5,
        )

        samples = self.sample(12 ** 2)
        save_image(
            samples.cpu().data,
            save_dir / f"sample_image-{self.current_epoch}.png",
            normalize=False,
            nrow=12,
            pad_value=0.5,
        )

        del test_input, recons, samples


if __name__ == '__main__':
    MyLightningModule(
        in_channels=1,
        latent_dim=1,
        hidden_dims=[1, 1],
        batch_size=32,
        lr=0.01,
        weight_decay=0,
    )
