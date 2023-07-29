from torch.utils.data import Dataset

import os
from os.path import join as ospj
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
c10 = [184,0,141] 
c11 = [184,134,0] 
c12 = [184,134,223]
label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

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
                if int(digit) < 0 or int(digit) > 12:
                    digit = 0
                row_pixels.append(digit)
                count += 1
        assert count==dim2, f"Actual count: {count}"
        affordance_pixels.append(row_pixels)
            
    # dimensions = [dim1, dim2]
    affordance_pixels = np.array(affordance_pixels, dtype=np.uint8)
    seg_map = label_colours[affordance_pixels]
    seg_map = np.array(seg_map, dtype=np.uint8)
    
    # print(seg_map.shape)

    # plt.imshow(seg_map)
    # plt.show()

    # Normalize the data to be in the range [0, 1]
    seg_map_normalized = seg_map.astype(np.float32) / 255.0

    # Convert the normalized 3D NumPy array to a PIL image
    seg_map_img = Image.fromarray((seg_map_normalized * 255).astype(np.uint8))
    return seg_map_img

def parse_object_labels(object_filename):
    with open(object_filename, "r") as f:
        object_labels = f.readlines()
        for i in range(len(object_labels)):
            object_labels[i] = object_labels[i].strip()
    return object_labels

class AffordanceDataset(Dataset):
    def __init__(self, root_dir, split_file, feature_extractor=None, transform=None):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.transform = transform

        # Read the split file (e.g., train_and_val.txt or val.txt) to get the list of image filenames
        with open(os.path.join(self.root_dir, split_file), "r") as f:
            self.image_filenames = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load RGB image
        image_filename = ospj(self.root_dir, "rgb", self.image_filenames[idx])
        image = Image.open(image_filename).convert("RGB")

        # Load Affordance Label
        affordance_file = self.image_filenames[idx].replace(".jpg", ".txt")
        affordance_filename = ospj(self.root_dir, "affordances_labels", affordance_file)
        affordances_labels = parse_affordance_labels(affordance_filename)

        # Load Object Label
        object_file = self.image_filenames[idx].replace(".jpg", ".txt")
        object_filename = ospj(self.root_dir, "object_labels", object_file)
        object_labels = parse_object_labels(object_filename)

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
            affordances_labels = self.transform(affordances_labels)

        # Extract features if feature_extractor is provided
        if self.feature_extractor is not None:
            encoded_input = self.feature_extractor(image, affordances_labels, return_tensors="pt")
        
        return {"image": image, "affordances_labels": affordances_labels, "objects_labels": object_labels, "encoded_input": encoded_input}
        
