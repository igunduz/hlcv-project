# train_eval.py

from torchvision.transforms import Compose, ToTensor, Resize
from dataset import SemanticSegmentationDataset
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch
import torch.nn as nn
import torch.optim as optim

# Set up the transform for image and affordances labels (resize and convert to tensor)
transform = Compose([Resize((128, 128)), ToTensor()])

# Create training and validation datasets
train_dataset = SemanticSegmentationDataset(root_dir="/icbb/projects/igunduz/hlcv/IIT_Affordances_2017",
                                            split_file="train_and_val.txt",
                                            transform=transform)
val_dataset = SemanticSegmentationDataset(root_dir="/icbb/projects/igunduz/hlcv/IIT_Affordances_2017",
                                          split_file="val.txt",
                                          transform=transform)

# Set up DataLoaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# load the SegFormer model
num_classes = 10  # 9 affordance classes + background
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Modify the output layer to match the number of classes 
model.decode_head.classifier = torch.nn.Conv2d(768, num_classes, kernel_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Set the number of epochs and iterate over the training data
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs = batch["image"]
        labels = batch["affordances"]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
