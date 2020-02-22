#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch.nn as nn
import torch
k=5 ##kernel size
class CNN(nn.Module):
  def __init__(self, charembedsize, m_word, f):
    super(CNN, self).__init__()
    self.conv1d=nn.Conv1d(in_channels=charembedsize, out_channels=f, kernel_size=k, stride=1, padding=0)
    self.relu=nn.ReLU()
    self.maxpool=torch.nn.MaxPool1d(m_word-k+1, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
  def forward(self, xreshaped):
    hidden1=self.conv1d(xreshaped)
    hidden2=self.relu(hidden1)
    xconv_out=torch.squeeze(self.maxpool(hidden2), -1)
    return xconv_out

### END YOUR CODE

