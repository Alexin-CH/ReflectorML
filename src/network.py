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

# ICNN: https://arxiv.org/abs/1609.07152

# class NonNegLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(out_features, in_features))
#         self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

#         torch.nn.init.xavier_uniform_(self.weight, gain=1e-3)

#     def forward(self, input):
#         W = torch.abs(self.weight)
#         return F.linear(input, W, self.bias)

# class ReCU(nn.Module):
#     def __init__(self):
#         super(ReCU, self).__init__()
#         self.activaction = nn.ReLU()

#     def forward(self, x):
#         return self.activaction(x) ** 3

# class ICNNLayer(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.linear = NonNegLinear(in_features, out_features, bias)
#         self.activation = nn.Softplus()
        
#     def forward(self, input):
#         x = self.linear(input)
#         return self.activation(x)

class NonNegLinear(nn.Linear):
    def forward(self, x):
        weight = self.weight.clamp(min=0)
        return F.linear(x, weight, self.bias)

class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.z_layers = nn.ModuleList()
        self.x_layers = nn.ModuleList()

        prev_h = 0
        for h in hidden_dims:
            self.z_layers.append(NonNegLinear(prev_h, h, bias=True))
            self.x_layers.append(nn.Linear(input_dim, h, bias=False))
            prev_h = h

        self.output_layer = nn.Linear(prev_h, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        z = None
        for i, (z_layer, x_layer) in enumerate(zip(self.z_layers, self.x_layers)):
            x_term = x_layer(x)
            if z is None:
                z = self.activation(x_term + z_layer(torch.zeros(x.size(0), 0, device=x.device)))
            else:
                z = self.activation(x_term + z_layer(z))
        return self.output_layer(z)

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
            ICNNLayer(256, 1),
        )

        # Initialize final layer to be very close to 0
        # This ensures we start with an almost perfect 45-degree planar mirror
        # This is very important to avoid divergence !
        with torch.no_grad():
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
