#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class Highway(nn.Module):


## class that map outputs from converlutional network to highway network.
 def __init__(self, emword_len):
    super(Highway, self).__init__()
    self.emword_len = emword_len
    self.projection = nn.Linear(emword_len, emword_len)
    self.gate = nn.Linear(emword_len, emword_len)
    self.relu = nn.ReLU()
    self.Sigmoid = nn.Sigmoid()


 def forward(self, Xconv_out):
    xproj = self.relu(self.projection(Xconv_out))
    xgate = self.Sigmoid(self.gate(Xconv_out))
    xhighway = xgate*xproj + (1 - xgate)*Xconv_out
    return xhighway

### END YOUR CODE 

