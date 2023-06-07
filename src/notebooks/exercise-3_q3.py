# %%
# Only for Colab, comment out if not using Colab
#from google.colab import drive
#drive.mount('/content/drive')

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


# %% [markdown]
# In Exercise 3, you will implement a convolutional neural network to perform image classification and explore methods to improve the training performance and generalization of these networks.
# We will use the CIFAR-10 dataset as a benchmark for our networks, similar to the previous exercise. This dataset consists of 50000 training images of 32x32 resolution with 10 object classes, namely airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The task is to implement a convolutional network to classify these images using the PyTorch library. The four questions are,
# 
# - Implementing a convolutional neural network, training it, and visualizing its weights (Question 1).
# - Experiment with batch normalization and early stopping (Question 2).
# - Data augmentation and dropout to improve generalization (Question 3).
# - Implement transfer learning from an ImageNet-pretrained model (Question 4).

# %% [markdown]
# ### Question 1: Implement Convolutional Network (10 points)
# 
# In this question, we will implement a five-layered convolutional neural network architecture as well as the loss function to train it. Refer to the comments in the code to the exact places where you need to fill in the code.
# 
# ![](../../data/exercise-3/fig1.png)

# %% [markdown]
# a) Our architecture is shown in Fig 1. It has five convolution blocks. Each block is consist of convolution, max pooling, and ReLU operation in that order. We will use 3×3 kernels in all convolutional layers. Set the padding and stride of the convolutional layers so that they maintain the spatial dimensions. Max pooling operations are done with 2×2 kernels, with a stride of 2, thereby halving the spatial resolution each time. Finally, stacking these five blocks leads to a 512 × 1 × 1 feature map. Classification is achieved by a fully connected layer. We will train convolutional neural networks on the CIFAR-10 dataset. Implement a class ConvNet to define the model described. The ConvNet takes 32 × 32 color images as inputs and has 5 hidden layers with 128, 512, 512, 512, 512 filters, and produces a 10-class classification. The code to train the model is already provided. Train the above model and report the training and validation accuracies. (5 points)
# 
# Please implement the above network (initialization and forward pass) in class `ConvNet` in `models/cnn/model.py`.
# 
# b) Implement the method `__str__` in `models/base_model.py`, which should give a string representaiton of the model. The string should show the number of `trainable` parameters for each layer. This gives us a measure of model capacity. Also at the end, the string contains the total number of trainable parameters for the entire model. (2 points)

# %% [markdown]
# ![](../../data/exercise-3/fig2.png)

# %% [markdown]
# c) Implement a function `VisualizeFilter` in `models/cnn/model.py`, which visualizes the filters of the first convolution layer implemented in Q1.a. In other words, you need to show 128 filters with size 3x3 as color images (since each filter has three input channels). Stack these into 3x3 color images into one large image. You can use the `imshow` function from the `matplotlib` library to visualize the weights. See an example in Fig. 2 Compare the filters before and after training. Do you see any patterns? (3 points). Please attach your output images before and after training in a cell with your submission.

# %%
from torchvision import transforms

from utils.parse_config import ConfigParser
from trainers.cnn_trainer import CNNTrainer
import data_loaders.data_modules as module_data

from copy import deepcopy
#%aimport -ConfigParser # Due to an issue of pickle and auto_reload

# %%
config = ConfigParser.wo_args(config='cfgs/exercise-3/cnn_cifar10.json', root_dir=PROJECT_ROOT)

# Get the preset names you want to use
preset_names = ['CIFAR10_WithFlip2', 'CIFAR10_WithFlip4', 
                'CIFAR10_WithFlip8','CIFAR10_WithColor',
                'CIFAR10_WithColorC','CIFAR10_WithColorPlus',
                'CIFAR10_WithColorB']  

# Loop over the preset names and train the model with each preset
for preset_name in preset_names:
    # Update the preset_name in the configuration
    config['data_module']['args']['preset_name'] = preset_name
    
    # Initialize the data module with the updated config
    datamodule = config.init_obj('data_module', module_data, root_dir=PROJECT_ROOT)
    
    # Get the data loaders for training and validation
    train_data_loader = datamodule.get_loader()
    valid_data_loader = datamodule.get_heldout_loader()
    
    # Test loader is the same as train loader, with modified arguments
    test_loader_args = deepcopy(config['data_module']['args'])
    test_loader_args['training'] = False
    test_loader_args['shuffle'] = False
    test_loader_args['heldout_split'] = 0.0
    
    # Initialize the test module with the modified config
    test_module = getattr(module_data, config['data_module']['type'])(root_dir=PROJECT_ROOT, **test_loader_args)
    # Get the loader from it
    test_loader = test_module.get_loader()
    
    # Initialize the trainer and train the model
    trainer_cnn = CNNTrainer(config=config, train_loader=train_data_loader, eval_loader=valid_data_loader)
    trainer_cnn.model.VisualizeFilter()
    trainer_cnn.train()
    trainer_cnn.model.VisualizeFilter()
