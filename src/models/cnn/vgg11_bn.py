import torchvision
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(self, layer_config, num_classes, activation, norm_layer, fine_tune, pretrained=True):
        super(VGG11_bn, self).__init__()

        # TODO: Initialize the different model parameters from the config file  #
        # for activation and norm_layer refer to cnn/model.py
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._layer_config = layer_config
        self._num_classes = num_classes
        self._activation = getattr(nn, activation["type"])(**activation["args"])
        self._norm_layer = norm_layer
        self._fine_tune = fine_tune
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._pretrained = pretrained
        self._build_model()

    def _build_model(self):
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the weights variable. Set it to None if  #
        # you want to train from scratch, 'DEFAULT'/'IMAGENET1K_V1' if you want to use  #
        # pretrained weights. You can either write it here manually or in config file   #
        # You can enable and disable training the feature extraction layers based on    # 
        # the fine_tune flag.                                                           #
        #################################################################################
        self.vgg11_bn = torchvision.models.vgg11_bn(pretrained=self._pretrained)
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.hidden1 = nn.Linear(25088, self._layer_config[0])
        self.hidden2 = nn.Linear(self._layer_config[0], self._layer_config[1])
        self.output = nn.Linear(self._layer_config[1], self._num_classes)

        self.vgg11_bn.classifier = nn.Sequential(
            self.hidden1,
            self._activation,
            getattr(nn, self._norm_layer["type"])(self._layer_config[0]),
            self.hidden2,
            self._activation,
            getattr(nn, self._norm_layer["type"])(self._layer_config[1]),
            self.output
        )

        if not self._fine_tune:
            for param in self.vgg11_bn.parameters():
                param.requires_grad = False
            for param in self.vgg11_bn.classifier.parameters():
                param.requires_grad = True        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.vgg11_bn(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x