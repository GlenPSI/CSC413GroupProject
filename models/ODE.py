import math
import torch
import torchcde
import pandas as pd
import numpy as np

######################
# This code is heavily based on implementation examples provided by the torchcde library as seen here: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
######################

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, args):
        ######################
        # input_channels is the number of input channels in the data X. 
        # hidden_channels is the number of channels for z_t.
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.device = torch.device(args.device)
        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, args, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.device = torch.device(args.device)
        self.func = CDEFunc(input_channels, hidden_channels, args)
        initial_layers = [torch.nn.Linear(input_channels, hidden_channels)]
        initial_layers.extend([torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.Tanh()) for _ in range(args.nLayers)])
        self.initial = torch.nn.Sequential(*initial_layers)
        output_layers = [torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.Tanh()) for _ in range(args.nLayers)]
        output_layers.append(torch.nn.Linear(hidden_channels, output_channels))
        self.readout = torch.nn.Sequential(*output_layers)
        self.interpolation = interpolation

    def forward(self, x):
        coeffs = torchcde.natural_cubic_coeffs(torch.from_numpy(x)).to(torch.float).to(self.device)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        
        ######################
        # Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y.squeeze(-1)