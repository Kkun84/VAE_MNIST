"""
Usage:
    train.py [--max_epochs=<int>] [--batch_size=<int>] [--num_workers=<int>] [--image_size=<int>][--latent_dim=<int>] [--hidden_dims=<int>...] [--lr=<float>] [--weight_decay=<float>] [--gpus=<int>] [--seed=<int>]

Options:
    -h --help           Show this screen.
    --max_epochs=<int>  Epoch num [default: 30].
    --batch_size=<int>  Batch size [default: 64].
    --num_workers=<int>  Num workers of DataLoader [default: 4].
    --image_size=<int>  Image width & height [default: 32].
    --latent_dim=<int>  Batch size [default: 64].
    --hidden_dims=<int>...
    --lr=<float>        Learning rate [default: 0.001].
    --weight_decay=<float> [default: 0].
    --gpus=<int>    Use GPU [default: 1].
    --seed=<int>  [default: 0].
"""

import pytorch_lightning as pl
from docopt import docopt
from pytorch_lightning import seed_everything
from torchinfo import summary

from lightning_module import MyDataModule, MyLightningModule


def main():
    args = docopt(__doc__)

    print(args)

    params = dict(
        max_epochs=int(args['--max_epochs']),
        image_size=int(args['--image_size']),
        latent_dim=args['--latent_dim'] and int(args['--latent_dim']),
        hidden_dims=(
            [16, 32, 64, 128]
            # [int(i) for i in args['--hidden_dims']] if args['--hidden_dims'] else None
        ),
        batch_size=int(args['--batch_size']),
        num_workers=int(args['--num_workers']),
        lr=args['--lr'] and float(args['--lr']),
        weight_decay=args['--weight_decay'] and float(args['--weight_decay']),
        gpus=int(args['--gpus']),
        seed=int(args['--seed']),
    )
    seed_everything(params['seed'])

    datamodule = MyDataModule(**params)

    params['in_channels'] = datamodule.channels
    params['train_data_size'] = 60000
    params['val_data_size'] = 10000

    params = {k: v for (k, v) in params.items() if v is not None}
    print(params)

    model = MyLightningModule(**params)
    summary(model)

    trainer = pl.Trainer(
        gpus=params['gpus'],
        max_epochs=params['max_epochs'],
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
