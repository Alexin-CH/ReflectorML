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
    """Learn the transport map's Jacobian field J_T directly (differential MA).

    forward(x) returns the (N, 2, 2) Jacobian of the transport map T at each
    point x, measured in the reflected (orientation-preserving) frame. The map
    itself is recovered by integrating this field along the ray from the origin,
    T(x) = int_0^1 J_T(s x) x ds, assuming T(0) = 0 in the source frame.

    We initialize the output to the identity Jacobian (det = +1) so the map is
    orientation-preserving and starts near T(x) = x, i.e. a planar-ish start.
    """
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            SineLayer(2, 512),
            SineLayer(512, 256),
            FINERLayer(256, 256),
            nn.Linear(256, 4),  # flattened (2, 2) Jacobian
        )

        # Init to the identity Jacobian: J = [[1,0],[0,1]] -> det = +1, T ~ x.
        with torch.no_grad():
            self.net[-1].weight.zero_()
            b = self.net[-1].bias
            b.zero_()
            b[0] = 1.0   # first row  (1, 0)
            b[3] = 1.0   # second row (0, 1)

    def forward(self, coords):
        out = self.net(coords)
        return out.view(-1, 2, 2)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, weights_only=True))
        return self
