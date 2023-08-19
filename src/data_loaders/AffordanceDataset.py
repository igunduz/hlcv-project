from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

import os
from os.path import join as ospj
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure the logging settings
logging.basicConfig(level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',  # Output log messages to a file (optional)
                    filemode='w')

logger = logging.getLogger()

background = [200, 222, 250]
c1 = [0,0,205]   
c2 = [34,139,34] 
c3 = [192,192,128]   
c4 = [165,42,42]    
c5 = [128,64,128]   
c6 = [204,102,0]  
c7 = [184,134,11] 
c8 = [0,153,153]
c9 = [0,134,141]
# c10 = [184,0,141] 
# c11 = [184,134,0] 
# c12 = [184,134,223]
label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9])
# , c10, c11, c12])

affordances = np.array(["contain", "cut", "display", "engine", "grasp", "hit", "pound", "support", "w-grasp"])

# Object
col0 = [0, 0, 0]
col1 = [0, 255, 255]
col2 = [255, 0, 255]
col3 = [0, 125, 255]
col4 = [55, 125, 0]
col5 = [255, 50, 75]
col6 = [100, 100, 50]
col7 = [25, 234, 54]
col8 = [156, 65, 15]
col9 = [215, 25, 155]
col10 = [25, 25, 155]

col_map = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]

def parse_affordance_labels(affordance_filename):
    with open(affordance_filename) as f:
        affordance_labels = f.readlines()
        dim1 = len(affordance_labels)
        dim2 = len(affordance_labels[0].split())
        
    count = 0
    # Creation of Affordance pixel grid
    # Checking image size to continue with creating Segmentation Map
    affordance_pixels = []
    for label in affordance_labels:
        row_pixels = []
        digits = label.split()
        count = 0
        for digit in digits:
            if digit.isdigit():
                if int(digit) < 0 or int(digit) > (len(label_colours) - 1):
                    digit = 0
                row_pixels.append(digit)
                count += 1
        assert count==dim2, f"Actual count: {count}"
        affordance_pixels.append(row_pixels)
            
    affordance_pixels = np.array(affordance_pixels, dtype=np.uint8)

    seg_map = label_colours[affordance_pixels]
    seg_map = torch.tensor(seg_map, dtype=torch.uint8).reshape(dim1, dim2, 3).permute(2, 0, 1)
    

    # Normalize the data to be in the range [0, 1]
    seg_map_normalized = seg_map / 255.0

    # Experimenting with Affordance pixel grid
    affordance_pixels = torch.from_numpy(affordance_pixels).reshape(dim1, dim2)
    # logger.info(f"Shape of affordance_pixels: {affordance_pixels.shape}")

    # Convert the normalized 3D NumPy array to a PIL image
    seg_map_img = transforms.ToPILImage()(seg_map_normalized)
    return seg_map_img, affordance_pixels

def parse_object_labels(object_filename):
    with open(object_filename, 'r') as file:
        lines = file.readlines()

    # Split each line into individual values and convert them to integers
    object_labels = [list(map(int, line.strip().split())) for line in lines]

    # Convert the data to a PyTorch tensor
    return torch.tensor(object_labels)

class AffordanceDataset(Dataset):
    def __init__(self, root_dir, split_file, feature_extractor=None, transform=None):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.transform = transform

        # Read the split file (e.g., train_and_val.txt or val.txt) to get the list of image filenames
        with open(os.path.join(self.root_dir, split_file), "r") as f:
            self.image_filenames = [line.strip() for line in f]

        self.id2label = {i:label for i, label in enumerate(affordances.tolist())}
        self.id2label[255] = "background"

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load RGB image
        image_filename = ospj(self.root_dir, "rgb", self.image_filenames[idx])
        image = Image.open(image_filename).convert("RGB")
        np_image = np.array(image)

        # Load Affordance Label
        affordance_file = self.image_filenames[idx].replace(".jpg", ".txt")
        affordance_filename = ospj(self.root_dir, "affordances_labels", affordance_file)
        affordance_rgb, affordance_labels = parse_affordance_labels(affordance_filename)

        # Load Object Label
        object_file = self.image_filenames[idx].replace(".jpg", ".txt")
        object_filename = ospj(self.root_dir, "object_labels", object_file)
        object_labels = parse_object_labels(object_filename)

        # Apply transformations if provided
        if self.transform is not None:
            tensor_image = self.transform(Image.fromarray(np_image))
            affordance_rgb = self.transform(affordance_rgb)

        pil_image = transforms.ToPILImage()(tensor_image).convert("RGB")
        pil_affordance_rgb = transforms.ToPILImage()(affordance_rgb).convert("RGB")

        # Extract features if feature_extractor is provided
        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(pil_image, affordance_labels, return_tensors="pt")
            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
        
        
        # return {"image": tensor_image, "affordance_rgb": affordance_rgb, "object_labels": object_labels, "encoded_input": encoded_inputs}
        return encoded_inputs
