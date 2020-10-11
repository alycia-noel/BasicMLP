# -*- coding: utf-8 -*-
"""
Homework One Network
Created on Thu Sep 24 11:34:04 2020

@author: ancarey
"""

from torch import nn
import torch.nn.functional as F 

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_layers)
        self.output = nn.Linear(hidden_layers, output_size)
        
    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = self.output(x)
    
        return x