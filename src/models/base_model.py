import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()

        #################################################################################
        # TODO: Q1.b) Print the number of trainable parameters for each layer and total number of trainable parameters
        # Simply update the ret_str by adding new lines to it.
        #################################################################################
        
        return ret_str