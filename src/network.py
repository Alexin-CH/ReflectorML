import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Improved fully-connected architecture (Wang, Peng, Perdikaris GRAD 2020.
# "Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural
# Networks", 2001.04536, eqs. 43-47). Multiplicative input interaction via two
# fixed feature branches U, V, with hidden states H_k interpolating between them:
#   U = phi(X W1 + b1),  V = phi(X W2 + b2)
#   H^(1)   = phi(X Wz,1 + bz,1)
#   Z^(k)   = phi(H^(k) Wz,k + bz,k), k=1..L
#   H^(k+1) = (1 - Z^(k)) .* U + Z^(k) .* V
#   f(x)    = H^(L+1) W + b

class MirrorSurfaceFV(nn.Module):
    def __init__(self, width=128, depth=4, activation=F.tanh):
        super().__init__()
        self.width = width
        self.depth = depth
        self.activation = activation
        d = 2  # input spatial dimension

        # Two fixed feature branches (multiplicative interaction)
        self.W1 = nn.Parameter(torch.Tensor(d, width))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.Tensor(d, width))
        self.b2 = nn.Parameter(torch.zeros(width))

        # Gate projections
        self.Wz = nn.ParameterList()
        self.bz = nn.ParameterList()
        for _ in range(depth):
            self.Wz.append(nn.Parameter(torch.Tensor(width, width)))
            self.bz.append(nn.Parameter(torch.zeros(width)))

        # First hidden projection
        self.Wz0 = nn.Parameter(torch.Tensor(d, width))
        self.bz0 = nn.Parameter(torch.zeros(width))

        # Output
        self.Wout = nn.Parameter(torch.Tensor(width, 1))
        self.bout = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        for w, b in [(self.W1, self.b1), (self.W2, self.b2),
                     (self.Wz0, self.bz0)] + \
                    [(w, b) for w, b in zip(self.Wz, self.bz)]:
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)
        nn.init.uniform_(self.Wout, -1e-8, 1e-8)
        nn.init.zeros_(self.bout)

    def forward(self, coords):
        U = self.activation(coords @ self.W1 + self.b1)
        V = self.activation(coords @ self.W2 + self.b2)
        H = self.activation(coords @ self.Wz0 + self.bz0)
        for w, b in zip(self.Wz, self.bz):
            Z = self.activation(H @ w + b)
            H = (1 - Z) * U + Z * V
        return H @ self.Wout + self.bout

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        return self


# SIREN: https://arxiv.org/abs/2006.09661

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=15):
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

class TanhLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=None):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return torch.tanh(self.linear(input))

class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            SineLayer(2, 512),
            SineLayer(512, 256),
            SineLayer(256, 256), # SineLayer or FINERLayer or HSineLayer
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
