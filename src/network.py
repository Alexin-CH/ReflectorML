import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x)

class FINERLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x * (1 + torch.abs(x)))

class HSineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = nn.Parameter(torch.tensor(2.0))

    def forward(self, input):
        x = torch.sinh(self.r * self.linear(input))
        return torch.sin(self.w0 * x)

class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            SineLayer(2, 512),
            SineLayer(512, 256),
            FINERLayer(256, 256), # SineLayer or FINERLayer or HSineLayer
            nn.Linear(256, 1)
        )

        self.init_weights()
        
    def init_weights(self):
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
            self.net[-1].weight.uniform_(-1e-8, 1e-8)

    def forward(self, coords):        
        # Neural offset (The freeform deformation)
        deformation = self.net(coords)
        return deformation

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        return self
