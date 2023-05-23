import os
from torchvision import datasets

from .base_data_modules import BaseDataModule
from utils.transform_presets import presets


class CIFAR10DataModule(BaseDataModule):
    """
    CIFAR10 data loading using BaseDataModule
    """
    def __init__(self, data_dir, preset_name, heldout_split=0.0, training=True, root_dir=None, **loader_kwargs):
        

        # Figure out the Transformation
        data_split = 'train' if training else 'eval'
        transform = presets[preset_name][data_split]
        print(f"transformations for split {data_split} are {transform}")

        data_dir = data_dir if root_dir is None else os.path.join(root_dir, data_dir)
        # Create the dataset!
        dataset = datasets.CIFAR10(root=data_dir, train=training, download=True, transform=transform)

        super().__init__(dataset, heldout_split=heldout_split, **loader_kwargs)