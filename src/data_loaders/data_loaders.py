from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader


class CIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        
        norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(root=self.data_dir, train=training, download=False, transform=norm_transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)