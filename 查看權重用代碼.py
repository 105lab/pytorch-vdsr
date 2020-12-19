# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:15:30 2020

@author: user
"""

# coding: utf-8
import torch
#from GRU_300 import GRU


# Load pre-trained model
model_a = torch.load(r'D:\git\pytorch-vdsr\model\model_epoch_50.pth').cpu()
model_a.eval()


# Display all model layer weights
for name, para in model_a.named_parameters():
    print('{}: {}'.format(name, para.shape))
    

