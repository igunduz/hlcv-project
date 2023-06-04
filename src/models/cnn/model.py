import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ######################################################################################################
        # TODO: Initialize the different model parameters from the config file                               #    
        # You can use the arguments given in the constructor. For activation and norm_layer                  #
        # to make it easier, you can use the following two lines                                             #                              
        #   self._activation = getattr(nn, activation["type"])(**activation["args"])                         #        
        #   self._norm_layer = getattr(nn, norm_layer["type"])                                               #
        # Or you can just hard-code using nn.Batchnorm2d and nn.ReLU as they remain fixed for this exercise. #
        ###################################################################################################### 
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.activation = getattr(nn, activation["type"])(**activation["args"])
        self.norm_layer = getattr(nn, norm_layer["type"])

        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._num_classes = num_classes
        self._drop_prob = drop_prob
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Convolutional Layers
        layers.append(nn.Conv2d(self._input_size, self._hidden_layers[0], kernel_size=3, padding=1))
        layers.append(getattr(nn, self._activation.__class__.__name__)())  # ReLU activation
        for i in range(len(self._hidden_layers) - 1):
            layers.append(nn.Conv2d(self._hidden_layers[i], self._hidden_layers[i + 1], kernel_size=3, padding=1))
            layers.append(getattr(nn, self._activation.__class__.__name__)())  # ReLU activation
            layers.append(getattr(nn, self._norm_layer.__class__.__name__)(self._hidden_layers[i + 1]))  # BatchNorm2d layer
            layers.append(nn.Dropout2d(self._drop_prob))  # Dropout layer

        # Output Layer
        layers.append(nn.Conv2d(self._hidden_layers[-1], self._num_classes, kernel_size=1))

        self.model = nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        raise NotImplementedError
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        x = self.model(x)
        x = x.mean(dim=(2, 3))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
