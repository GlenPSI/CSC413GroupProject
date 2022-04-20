from torch import nn
import torch

import numpy as np
import math

### Code from library : https://github.com/ojus1/Time2Vec-PyTorch
### Paper https://arxiv.org/pdf/1907.05321.pdf

def t2v(tau, f, hidden_size, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)



class SineActivation(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(SineActivation, self).__init__()
        self.hidden_size = hidden_size
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, hidden_size-1))
        self.b = nn.parameter.Parameter(torch.randn(hidden_size-1))
        self.f = torch.sin

    def forward(self, tau):
        # out in_feature x hidden_size
        return t2v(tau, self.f, self.hidden_size, self.w, self.b, self.w0, self.b0)



class CosineActivation(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(CosineActivation, self).__init__()
        self.hidden_size = hidden_size
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, hidden_size-1))
        self.b = nn.parameter.Parameter(torch.randn(hidden_size-1))
        self.f = torch.cos

    def forward(self, tau):
        # out in_feature x hidden_size
        return t2v(tau, self.f, self.hidden_size, self.w, self.b, self.w0, self.b0)



class Time2Vector(nn.Module):
    def __init__(self, input_size, hidden_size, activation="sin"):
        super(Time2Vector, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(input_size, hidden_size)
        else: # cos
            self.l1 = CosineActivation(input_size, hidden_size)


    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.l1(x)
        return x
    
    
    
if __name__ == "__main__":
    sineact = SineActivation(5, 256)
    cosact = CosineActivation(5, 256)

    print(sineact(torch.rand(15,5)).shape)
    print(cosact(torch.rand(15,5)).shape)