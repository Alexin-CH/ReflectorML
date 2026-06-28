import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import icnn

# SIREN: https://arxiv.org/abs/2006.09661

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            limit = 1 / self.linear.weight.shape[1]
            self.linear.weight.uniform_(-limit, limit)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x)

# FINER: https://arxiv.org/abs/2312.02434

class FINERLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            limit = np.sqrt(6 / self.linear.weight.shape[1]) / self.w0
            self.linear.weight.uniform_(-limit, limit)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x * (1 + torch.abs(x)))

# H-SIREN: https://arxiv.org/abs/2410.04716

class HSineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = nn.Parameter(torch.tensor(2.0))

    def forward(self, input):
        x = torch.sinh(self.r * self.linear(input))
        return torch.sin(self.w0 * x)

#
# #
# # #
# #
#

class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            SineLayer(2, 512),
            SineLayer(512, 256),
            FINERLayer(256, 256),
            nn.Linear(256, 1),
            # icnn.ICNN(64, 1, 1)
        )

        # Initialize final layer to be very close to 0
        # This ensures we start with an almost perfect 45-degree planar mirror
        # This is very important to avoid divergence !
        with torch.no_grad():
            # self.net[-1].backbone.W_out.weight.normal_(0, 1e-8)
            limit = 1e-8
            self.net[-1].weight.uniform_(-limit, limit)

    def forward(self, coords):        
        # Neural offset (The freeform deformation)
        deformation = self.net(coords)
        return deformation

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, weights_only=True))
        return self
