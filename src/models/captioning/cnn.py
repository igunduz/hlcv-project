import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """
        Load a pretrained ResNet-152 and modify top layers to extract features
        """
        super(EncoderCNN, self).__init__()
        
        resnet = torchvision.models.resnet152(pretrained=True)
        #########################
        # TODO 
        # Create a sequential model with all the layers of resnet except the last fc layer.
        # Add a linear layer to bring resnet features down to embed_size
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        #########################
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        #########################
        # TODO 
        # Run your input images through the modules you created above
        # Make sure to freeze the weights of the resnet layers
        # finally return the normalized features
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return F.normalize(features, dim=1)
        #########################