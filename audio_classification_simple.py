# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:28:17 2023

@author: bmkea
"""

import torch
from torch import nn


# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(344, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x)
        x = self.fc2(x)
       #print(x.shape)
        return x

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device