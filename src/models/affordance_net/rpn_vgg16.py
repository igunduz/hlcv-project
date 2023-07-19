
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(cfg):
    """Generating rpn model for given hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        rpn_model = torch.nn.Module
        feature_extractor = feature extractor layer from the base model
    """
    base_model = models.vgg16(pretrained=True)
    base_model.features[0].requires_grad_(False)
    base_model.features[1].requires_grad_(False)
    base_model.features[2].requires_grad_(False)
    base_model.features[3].requires_grad_(False)
    feature_extractor = base_model.features[23]
    l2_reg = torch.nn.Regulizer(cfg.WEIGHT_DECAY)

    rpn_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    rpn_cls = nn.Conv2d(512, cfg.ANCHOR_COUNT, kernel_size=1, stride=1)
    rpn_reg = nn.Conv2d(512, cfg.ANCHOR_COUNT * 4, kernel_size=1, stride=1)
    rpn_model = nn.Sequential(rpn_conv, nn.ReLU(inplace=True), rpn_cls, rpn_reg)
    return rpn_model, feature_extractor

def init_model(model):
    """Initializing model with dummy data for loading weights with optimizer state and also graph construction.
    inputs:
        model = torch.nn.Module
    """
    model(torch.randn(1, 3, 500, 500))

