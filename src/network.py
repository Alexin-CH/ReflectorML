import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FiLMSineLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta  = nn.Parameter(torch.zeros(out_features))

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            limit = np.sqrt(6 / self.linear.weight.shape[1]) / self.omega_0
            self.linear.weight.uniform_(-limit, limit)

    def forward(self, input):
        x = self.gamma * self.linear(input) + self.beta
        return torch.sin(self.omega_0 * x)

class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            FiLMSineLinearLayer(2, 512, omega_0=40),
            FiLMSineLinearLayer(512, 256, omega_0=30),
            FiLMSineLinearLayer(256, 256, omega_0=20),
            FiLMSineLinearLayer(256, 256, omega_0=10),
            nn.Linear(256, 1)
        )
        
        # Initialize final layer to be very close to 0
        # This ensures we start with a perfect 45-degree planar mirror
        with torch.no_grad():
            self.net[-1].weight.uniform_(-1e-8, 1e-8)
            self.net[-1].bias.zero_()

    def forward(self, coords):        
        # Neural offset (The freeform deformation)
        deformation = self.net(coords)
        return deformation
