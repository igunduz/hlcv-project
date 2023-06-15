import torch
import torchvision
import torch.nn as nn


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
        raise NotImplementedError
        #########################
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        #########################
        # TODO 
        # Run your input images through the modules you created above
        # Make sure to freeze the weights of the resnet layers
        # finally return the normalized features
        raise NotImplementedError
        #########################