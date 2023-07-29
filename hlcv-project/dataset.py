# dataset.py

from torch.utils.data import Dataset
from PIL import Image
import os

class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read the split file (e.g., train_and_val.txt or val.txt) to get the list of image filenames
        with open(os.path.join(self.root_dir, split_file), "r") as f:
            self.image_filenames = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load RGB image
        image_filename = os.path.join(self.root_dir, "rgb", self.image_filenames[idx] + ".jpg")
        image = Image.open(image_filename)

        # Load affordances labels
        affordances_filename = os.path.join(self.root_dir, "affordances_labels", self.image_filenames[idx] + ".png")
        affordances = Image.open(affordances_filename)

        # Load object labels
        object_labels_filename = os.path.join(self.root_dir, "object_labels", self.image_filenames[idx] + ".txt")
        with open(object_labels_filename, "r") as f:
            object_labels = [line.strip().split() for line in f]

        # TODO Implement any necessary preprocessing (e.g., resizing, data augmentation) 

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
            affordances = self.transform(affordances)

        return {"image": image, "affordances": affordances, "object_labels": object_labels}
