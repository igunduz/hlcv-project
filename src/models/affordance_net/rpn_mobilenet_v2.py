import torch
import torch.nn as nn
import torchvision.models as models

def get_model(hyper_params):
    """Generating rpn model for given hyper params.
    Inputs:
        hyper_params = dictionary
    Outputs:
        rpn_model = torch.nn.Module
        feature_extractor = feature extractor layer from the base model
    """
    img_size = hyper_params["img_size"]
    base_model = models.mobilenet_v2(pretrained=True)
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
    rpn_conv = nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1)
    rpn_cls = nn.Conv2d(512, hyper_params["anchor_count"], kernel_size=1, stride=1)
    rpn_reg = nn.Conv2d(512, hyper_params["anchor_count"] * 4, kernel_size=1, stride=1)
    rpn_model = nn.Sequential(feature_extractor, rpn_conv, nn.ReLU(inplace=True), rpn_cls, rpn_reg)
    return rpn_model, feature_extractor

def init_model(model):
    """Initializing model with dummy data for loading weights with optimizer state and also graph construction.
    Inputs:
        model = torch.nn.Module
    """
    model(torch.randn(1, 3, 500, 500))
