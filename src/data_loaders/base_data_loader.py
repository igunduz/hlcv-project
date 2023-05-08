from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        
        self.dataset = dataset
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(self.dataset)

        self.train_dataset, self.valid_dataset = self._split_data(self.validation_split)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(dataset=self.train_dataset, **self.init_kwargs)

    def _split_data(self, split):
        if split == 0.0:
            return self.dataset, None
        
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        
        len_train = self.n_samples - len_valid
        mask = list(range(len_train))
        train_dataset = Subset(self.dataset, mask)
        mask = list(range(len_train, len_train + len_valid))
        valid_dataset = Subset(self.dataset, mask)

        self.n_samples = len(train_dataset)

        return train_dataset, valid_dataset

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(dataset=self.valid_dataset, **self.init_kwargs)
