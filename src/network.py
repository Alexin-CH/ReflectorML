import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# SIREN: https://arxiv.org/abs/2006.09661

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x)

# FINER: https://arxiv.org/abs/2312.02434

class FINERLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

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


class MirrorSurface(nn.Module):
    """Learn the transport map's Jacobian field J_T directly (differential MA).

    forward(x) returns the (N, 2, 2) Jacobian of the transport map T at each
    point x, measured in the reflected (orientation-preserving) frame. The map
    itself is recovered by integrating this field along the ray from the origin,
    T(x) = int_0^1 J_T(s x) x ds, assuming T(0) = 0 in the source frame.

    J_T is parameterized as symmetric positive-definite (SPD) via its Cholesky
    factor: J = L L^T with L = [[p, 0], [q, r]]. This is the exact function
    class of the Brenier map T = grad(phi) (phi convex), so det J = p^2 r^2 >= 0
    is guaranteed by construction and the map is always a valid, fold-free
    orientation-preserving gradient field.
    """
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            SineLayer(2, 512),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 3),  # Cholesky entries (p, q, r)
        )

        # Init to the identity Jacobian: p=1, q=0, r=1 -> J = I, det = +1,
        # T ~ x. A planar-ish start in the reflected frame. With exp() on p,r,
        # identity corresponds to raw outputs p=0, r=0 (and q=0 by the zero
        # weight init).
        with torch.no_grad():
            limit = 1 / self.net[0].linear.weight.shape[1]
            self.net[0].linear.weight.uniform_(-limit, limit)

            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, coords):
        p, q, r = self.net(coords).chunk(3, dim=1)  # (N,1) each
        p = p.exp()
        r = r.exp()

        # L = [[p, 0], [q, r]] ; J = L L^T (SPD by construction)
        j11 = p * p
        j12 = p * q
        j22 = q * q + r * r

        j = torch.cat([
            torch.cat([j11, j12], dim=1),
            torch.cat([j12, j22], dim=1),
        ], dim=1).view(-1, 2, 2)
        return j

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, weights_only=True))
        return self
