from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from .cbam import *

class Resnet(nn.Module):
    def __init__(self, num_classes=7,pretrained=False):
        super(Resnet, self).__init__()
        
        resnet = models.resnet18(pretrained)        
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])               
        
        self.cbam = CBAM(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)             
        self.bn = nn.BatchNorm1d(num_classes) 
    
    def forward(self, x):
        x = self.features(x) 
        out = self.cbam(x)         
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)                   
        out = self.bn(out)

        return out
        

