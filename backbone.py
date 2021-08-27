import torch
import torchvision
from torch import nn

def get_resnet18(pre_imgnet = False, num_classes = 64):
    
    model = torchvision.models.resnet18(pretrained = pre_imgnet)
    model.fc = nn.Linear(512, 64)
    return model
