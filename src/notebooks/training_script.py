# %%
# Only for Colab, comment out if not using Colab
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# Change this line if you're using Colab to something like '/content/drive/MyDrive/TeamX/'
# where TeamX is just the clone of repository on your Google Drive
# and you have mounted the drive at /content/drive  
# See the Tutorial Slides for more detail.

# Works on your local machine but not on Colab!
PROJECT_ROOT = '../..' 

# Fix this path and use this one on Colab
# PROJECT_ROOT = '/content/drive/MyDrive/TeamX' 

# %%
import sys
from os.path import join as ospj

sys.path.append(ospj(PROJECT_ROOT, 'src'))

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Playing around with the data
# ### 1. Load the data
# Using the IIT-AFF dataset, we will load the data and see what it looks like.
# ### 2. Visualize the data
# We will visualize the data to see what it looks like.
# ### 3. Preprocess the data
# Create a DataLoaders object for the data.

# %% [markdown]
# # Converting Text File to Segmentation Map

# %%
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
# import cv2

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import torchvision.transforms as transforms
from transformers import SegformerFeatureExtractor
from data_loaders.AffordanceDataset import AffordanceDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Resize



# data_dir = ospj(PROJECT_ROOT, 'data', 'IIT_Affordances_2017')

# background = (200, 222, 250)
# c1 = (0,0,205)   
# c2 = (34,139,34) 
# c3 = (192,192,128)   
# c4 = (165,42,42)    
# c5 = (128,64,128)   
# c6 = (204,102,0)  
# c7 = (184,134,11) 
# c8 = (0,153,153)
# c9 = (0,134,141)
# c10 = (184,0,141) 
# c11 = (184,134,0) 
# c12 = (184,134,223)
# label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

# # Object
# col0 = (0, 0, 0)
# col1 = (0, 255, 255)
# col2 = (255, 0, 255)
# col3 = (0, 125, 255)
# col4 = (55, 125, 0)
# col5 = (255, 50, 75)
# col6 = (100, 100, 50)
# col7 = (25, 234, 54)
# col8 = (156, 65, 15)
# col9 = (215, 25, 155)
# col10 = (25, 25, 155)

# col_map = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]


# # img = Image.open(ospj(data_dir, 'rgb', '00_00000090.jpg'))
# # plt.imshow(img)

# # %%
# def parse_affordance_labels(affordance_filename):
#     with open(affordance_filename) as f:
#         affordance_labels = f.readlines()
#         dim1 = len(affordance_labels)
#         dim2 = len(affordance_labels[0].split())
        
#     count = 0
#     # Creation of Affordance pixel grid
#     # Checking image size to continue with creating Segmentation Map
#     affordance_pixels = []
#     for label in affordance_labels:
#         row_pixels = []
#         digits = label.split()
#         count = 0
#         for digit in digits:
#             if digit.isdigit():
#                 if int(digit) < 0 or int(digit) > (len(label_colours) - 1):
#                     digit = 0
#                 row_pixels.append(digit)
#                 count += 1
#         assert count==dim2, f"Actual count: {count}"
#         affordance_pixels.append(row_pixels)
            
#     # dimensions = [dim1, dim2]
#     affordance_pixels = np.array(affordance_pixels, dtype=np.uint8)
#     # for i in range(dim1):
#     #     for j in range(dim2):
#     #         if affordance_pixels[i][j] < 0 or affordance_pixels[i][j] > 12:
#     #             logger.info(f"Negative Value: {affordance_pixels[i][j]}")

#     seg_map = label_colours[affordance_pixels]
#     seg_map = torch.tensor(seg_map, dtype=torch.uint8).reshape(dim1, dim2, 3).permute(2, 0, 1)
    
#     # print(seg_map.shape)

#     # plt.imshow(seg_map)
#     # plt.show()

#     # Normalize the data to be in the range [0, 1]
#     seg_map_normalized = seg_map / 255.0

#     # Convert the normalized 3D NumPy array to a PIL image
#     seg_map_img = transforms.ToPILImage()(seg_map_normalized)
#     return seg_map_img

# def parse_object_labels(object_filename):
#     with open(object_filename, 'r') as file:
#         lines = file.readlines()

#     # Split each line into individual values and convert them to integers
#     object_labels = [list(map(int, line.strip().split())) for line in lines]

#     # Convert the data to a PyTorch tensor
#     return torch.tensor(object_labels)

# # %%

# segformer_b0_feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# # segformer_b0_feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
# segformer_b0_feature_extractor.reduce_labels = True
# segformer_b0_feature_extractor.size = 600

# # %%

# transform = transforms.Compose([
#     # transforms.Resize((600,600)),
#     transforms.ToTensor(),
# ])

# image_filename = ospj(data_dir, "rgb", "ILSVRC2014_train_00044663.jpg")
# image = Image.open(image_filename).convert("RGB")
# np_image = np.array(image)

# # Load Affordance Label
# affordance_file = "ILSVRC2014_train_00044663.jpg".replace(".jpg", ".txt")
# affordance_filename = ospj(data_dir, "affordances_labels", affordance_file)
# affordances_labels = parse_affordance_labels(affordance_filename)

# # Load Object Label
# object_file = "ILSVRC2014_train_00044663.jpg".replace(".jpg", ".txt")
# object_filename = ospj(data_dir, "object_labels", object_file)
# object_labels = parse_object_labels(object_filename)

# # Apply transformations if provided
# if transform is not None:
#     tensor_image = transform(Image.fromarray(np_image))
#     affordances_labels = transform(affordances_labels)

# tensor_image = transforms.ToPILImage()(tensor_image).convert("RGB")
# plt.imshow(tensor_image)
# plt.show()

# affordances_labels = transforms.ToPILImage()(affordances_labels).convert("RGB")
# plt.imshow(affordances_labels)
# plt.show()


# # Extract features if feature_extractor is provided
# if segformer_b0_feature_extractor is not None:
#     encoded_inputs = segformer_b0_feature_extractor(tensor_image, affordances_labels, return_tensors="pt")
#     for k,v in encoded_inputs.items():
#         encoded_inputs[k].squeeze_()

# # %%
# for k, v in encoded_inputs.items():
#     print(f"{k}: {v.shape}")

# # %% [markdown]
# # # Experimenting with the DataLoader

# %%


def collate_fn(batch):
    images = [item["image"] for item in batch]
    affordances_labels = [item["affordances_labels"] for item in batch]
    object_labels = [item["object_labels"] for item in batch]
    encoded_inputs = [item["encoded_input"] for item in batch]

    # Pad images to the same size
    max_width = max(image.shape[-1] for image in images)
    max_height = max(image.shape[-2] for image in images)

    padded_images = [
        F.pad(image, (0, max_width - image.shape[-1], 0, max_height - image.shape[-2]))
        for image in images
    ]
    padded_images = torch.stack(padded_images)

    # # Pad affordances labels to the same size
    max_width = max(label.shape[-1] for label in affordances_labels)
    max_height = max(label.shape[-2] for label in affordances_labels)
    padded_affordances_labels = [
        F.pad(label, (0, max_width - label.shape[-1], 0, max_height - label.shape[-2]))
        for label in affordances_labels
    ]
    padded_affordances_labels = torch.stack(padded_affordances_labels)

    # Pad objects labels to the same size
    max_length = max(label.shape[0] for label in object_labels)
    padded_object_labels = [
        F.pad(label, (0, 0, 0, max_length - label.shape[0]))
        for label in object_labels
    ]

    return {
        "image": padded_images,
        "affordances_labels": padded_affordances_labels,
        "object_labels": torch.stack(padded_object_labels),
        "encoded_input": encoded_inputs
    }

# %%

data_dir = ospj(PROJECT_ROOT, 'data', 'IIT_Affordances_2017')
log_dir = ospj(PROJECT_ROOT, 'saved', 'log')

tb_logger = pl.loggers.TensorBoardLogger(log_dir)

segformer_b0_feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# segformer_b0_feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
segformer_b0_feature_extractor.reduce_labels = True
segformer_b0_feature_extractor.size = 600


transform = Compose([Resize((600,600)),
                     ToTensor()])

# example_dataset = AffordanceDataset(root_dir=data_dir,
#                                 split_file="example.txt",
#                                 feature_extractor=segformer_b0_feature_extractor,
#                                 transform=transform)

train_dataset = AffordanceDataset(root_dir=data_dir,
                                split_file="train_and_val.txt",
                                feature_extractor=segformer_b0_feature_extractor,
                                transform=transform)
                    

validation_dataset = AffordanceDataset(root_dir=data_dir, 
                                  split_file="val.txt",
                                  feature_extractor=segformer_b0_feature_extractor, 
                                  transform=transform)

test_dataset = AffordanceDataset(root_dir=data_dir,
                                split_file="test.txt",
                                feature_extractor=segformer_b0_feature_extractor,
                                transform=transform)

batch_size = 32

# example_loader = DataLoader(example_dataset, batch_size=batch_size, shuffle=True, num_workers=2) #, collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4) #, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4) #, collate_fn=collate_fn)

# %%

torch.cuda.empty_cache()

# %%
from trainers.segformer_trainer1 import SegformerFinetuner


segformer_finetuner = SegformerFinetuner(
    validation_dataset.id2label, 
    train_dataloader=train_loader, 
    val_dataloader=validation_loader, 
    test_dataloader=test_loader, 
    metrics_interval=1,
)

# %%

early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=3, 
    verbose=False, 
    mode="min",
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

trainer = pl.Trainer(
    accelerator="gpu",
    logger=tb_logger,
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=5,
    val_check_interval=len(validation_loader),
)

# Train the model
trainer.fit(segformer_finetuner)

# Test the model
results = trainer.test(ckpt_path="best")


