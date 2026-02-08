import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# --- THE SURFACE NETWORK (SIREN) ---
# We use a Sine-based MLP because we need accurate 2nd derivatives (Curvature)
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            limit = np.sqrt(6 / self.linear.weight.shape[1]) / self.omega_0
            self.linear.weight.uniform_(-limit, limit)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (x, z) coordinates on the aperture plane
        # Output: y height deviation from the base 45-degree plane
        
        self.net = nn.Sequential(
            SineLayer(2, 64, omega_0=30),
            SineLayer(64, 64, omega_0=20),
            SineLayer(64, 64, omega_0=10),
            SineLayer(64, 64, omega_0=10),
            nn.Linear(64, 1)
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
