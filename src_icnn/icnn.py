import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ICNN: https://arxiv.org/abs/1609.07152

class ConvexBackbone(nn.Module):
    """
    Generic convex backbone with configurable activation:
        z1 = act(W1 x)
        z_l = act(W_l x + U_{l-1} z_{l-1}), U>=0
        out = W_out x + U_out z_last, U_out>=0
    """
    def __init__(self, input_dim, hidden_dim, depth, activation, residual_scale=1.0):
        super().__init__()
        assert depth >= 1

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.activation = activation
        self.residual_scale = residual_scale

        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_layers = nn.ModuleList()
        self.U_layers_raw = nn.ParameterList()

        for _ in range(depth - 1):
            self.W_layers.append(nn.Linear(input_dim, hidden_dim))
            u = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
            nn.init.constant_(u, -5.0)
            self.U_layers_raw.append(u)

        self.W_out = nn.Linear(input_dim, 1)
        self.U_out_raw = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.constant_(self.U_out_raw, -5.0)

        # Near-planar/identity start so initial mapped rays hit the target
        # support (non-empty support_mask => active MA) WHILE keeping the
        # hidden layers trainable. softplus(-5)~6.7e-3 gives a small nonzero
        # U_out: output starts small but gradients still propagate to W_in/U.
        # A fully-zeroed U_out (-30, softplus~9e-14) keeps MA active but kills
        # backprop to the backbone (~1e-13 gradients => the net never learns).
        with torch.no_grad():
            self.W_out.weight.uniform_(-1e-8, 1e-8)
            self.W_out.bias.uniform_(-1e-8, 1e-8)
            nn.init.constant_(self.U_out_raw, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.W_in(x))
        for w, u_raw in zip(self.W_layers, self.U_layers_raw):
            u = torch.nn.functional.softplus(u_raw)
            z = z + self.residual_scale * self.activation(w(x) + z @ u.T)

        u_out = torch.nn.functional.softplus(self.U_out_raw)
        return self.W_out(x) + z @ u_out.T


class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth,
                 activation, residual_scale=1.0):
        super().__init__()
        self.backbone = ConvexBackbone(
            input_dim,
            hidden_dim,
            depth,
            activation,
            residual_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
