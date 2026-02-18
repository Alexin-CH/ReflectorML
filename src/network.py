import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x)

class HSirenLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        x = torch.sinh(2 * self.linear(input))
        return torch.sin(self.w0 * x)

class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            SirenLayer(2, 512),
            SirenLayer(512, 256),
            SirenLayer(256, 256),
            HSirenLayer(256, 1)
        )
        
        with torch.no_grad():
            # Siren first layer initialization
            limit = 1 / self.net[0].linear.weight.shape[1]
            self.net[0].linear.weight.uniform_(-limit, limit)

            # Siren non-first layers initialization
            for layer in self.net[1:-1]:
                limit = np.sqrt(6 / layer.linear.weight.shape[1]) / layer.w0
                layer.linear.weight.uniform_(-limit, limit)

            # Initialize final layer to be very close to 0
            # This ensures we start with an almost perfect 45-degree planar mirror
            # This is very important to avoid divergence !
            self.net[-1].linear.weight.uniform_(-1e-8, 1e-8)

    def forward(self, coords):        
        # Neural offset (The freeform deformation)
        deformation = self.net(coords)
        return deformation
