import torch.nn as nn
import torch

from ..base_model import BaseModel


class MultiLayerPerceptron(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, drop_prob=0.0):
        super(MultiLayerPerceptron, self).__init__()
        
        # TODO: Initialize the different model parameters from the config file  #
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.drop_prob = drop_prob

        self.build_model()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        # Make use of linear and relu layers from the torch.nn module                   #
        #################################################################################

        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layers.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        for i, units in enumerate(self.hidden_layers):
            layers.append(nn.ReLU()) if self.activation == 'ReLU' else layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=self.drop_prob))
            if i == len(self.hidden_layers)-1:
                layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))
            else:
                layers.append(nn.Linear(units, self.hidden_layers[i+1]))
        self.layers = nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        # Note that you do not need to use the softmax operation at the end.            #
        # Softmax is only required for the loss computation and the criterion will take #
        # care of it later.                                                             #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = torch.flatten(x, 1)
        x = self.layers(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
    