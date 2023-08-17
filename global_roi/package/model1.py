from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from .resnet_cbam1  import resnet18 as resnet_cbam1
from .resnet_cbam2  import resnet18 as resnet_cbam2
from .cbam import *
from torchvision import models
from .transformer_layer  import *

class ResNet(nn.Module):
    def __init__(self, num_classes=8, n_head=8, resnet_cbam=1):
        super(ResNet, self).__init__()
        
        ''' whole '''
        if resnet_cbam==1:
            resnet_whole = resnet_cbam1(use_cbam=True, num_classes=num_classes)
        elif resnet_cbam ==2:
            resnet_whole = resnet_cbam2(use_cbam=True, num_classes=num_classes)             
        
        self.features_whole = nn.Sequential(*list(resnet_whole.children())[:-3])               
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ''' roi '''    
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(
            *list(resnet.children())[0:3], #7*7,64->bn->relu            
            *list(resnet.children())[4:-4], #conv2-3            
            my_Block_2(n_embd=128,n_head=n_head),                      
            *list(resnet.children())[-4:-3], #conv4
            my_Block_2(n_embd=256,n_head=n_head),
            *list(resnet.children())[-3:-2], #conv5
            )        
                
        self.relu = nn.ReLU(True)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))        
        self.fc_roi = nn.Linear(512*3, 512)        
        self.bn_roi = nn.BatchNorm1d(512) 
        
        # self.channelGate = ChannelGate(512*3)

        ''' whole, roi cat'''
        self.fc = nn.Linear(512, num_classes)        
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, data, eye, nose, mouth):
        ''' whole '''
        whole = self.features_whole(data)
        whole = self.avgpool(whole)
        whole = torch.flatten(whole, 1) #512
        
        x_whole = self.fc(whole)
        x_whole = self.bn(x_whole)

        ''' roi '''            
        # roi 特征提取
        eye_data = self.features(eye)       
        nose_data = self.features(nose)       
        mouth_data = self.features(mouth)

        # roi cat 512*3
        roi = torch.cat((eye_data,nose_data,mouth_data),1)        
        # roi = self.channelGate(roi)
        
        roi = self.maxpool(roi)
        roi = torch.flatten(roi, 1)

        roi = self.relu(roi)  
        roi = self.fc_roi(roi)
        roi = self.bn_roi(roi)        #512

        x_roi = self.fc(roi)
        x_roi = self.bn(x_roi)

        ''' whole, roi cat'''
        x = whole+roi
        x = self.relu(x) 
        x = self.fc(x)
        x = self.bn(x)

        return x, x_whole, x_roi