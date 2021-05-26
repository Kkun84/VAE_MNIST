from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms.transforms import Lambda


class MyDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        image_size: int,
        batch_size: int,
        num_workers: int,
        data_dir: Union[str, Path] = 'data',
        pin_memory: bool = True,
        **_
    ):
        super().__init__()

        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[0:1]),
                # transforms.Normalize(0.5, 0.5),
            ]
        )
        return

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def channels(self) -> int:
        return 1

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        MNIST(
            root=str(self.data_dir / ''),
            train=False,
            download=True,
            transform=self.transform,
        )
        MNIST(
            root=str(self.data_dir / ''),
            train=False,
            download=True,
            transform=self.transform,
        )
        return

    def setup(self, stage: Optional[str] = None) -> None:
        assert stage is None or stage in ['fit', 'test'], stage
        self.data_train = MNIST(
            str(self.data_dir), train=True, transform=self.transform
        )
        self.data_val = MNIST(str(self.data_dir), train=False, transform=self.transform)
        return

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         shuffle=False,
    #     )


if __name__ == '__main__':

    image_size = 32
    batch_size = 64
    num_workers = 4

    datamodule = MyDataModule(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.prepare_data()
    datamodule.setup()
