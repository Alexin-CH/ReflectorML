import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ICNN: https://arxiv.org/abs/1609.07152
# SOC-ICNN: https://arxiv.org/pdf/2604.22355

class ReCU(nn.Module):
    def __init__(self):
        super(ReCU, self).__init__()
        self.activaction = nn.ReLU()

    def forward(self, x):
        return self.activaction(x) ** 2

class ConvexBackbone(nn.Module):
    """
    Generic convex backbone with configurable activation:
        z1 = act(W1 x)
        z_l = act(W_l x + U_{l-1} z_{l-1}), U>=0
        out = W_out x + U_out z_last, U_out>=0
    """
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        assert depth >= 1

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.activation = ReCU()

        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_layers = nn.ModuleList()
        self.U_layers_raw = nn.ParameterList()

        for _ in range(depth - 1):
            self.W_layers.append(nn.Linear(input_dim, hidden_dim))
            self.U_layers_raw.append(nn.Parameter(torch.zeros(hidden_dim, hidden_dim)))

        self.W_out = nn.Linear(input_dim, 1)
        self.U_out_raw = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.W_in(x))
        for w, u_raw in zip(self.W_layers, self.U_layers_raw):
            u = torch.nn.functional.softplus(u_raw)
            z = self.activation(w(x) + z @ u.T)

        u_out = torch.nn.functional.softplus(self.U_out_raw)
        return self.W_out(x) + z @ u_out.T


class ICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class QuadICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, quad_rank: int, num_quad_blocks: int = 1):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth)
        self.num_quad_blocks = num_quad_blocks
        self.Ls = nn.Parameter(torch.randn(num_quad_blocks, input_dim, quad_rank) * 0.01)
        self.alpha_raw = nn.Parameter(torch.zeros(num_quad_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        quad_terms = []
        alpha = torch.nn.functional.softplus(self.alpha_raw) + 1e-8
        for h in range(self.num_quad_blocks):
            q = x @ self.Ls[h]
            quad_terms.append(0.5 * alpha[h] * torch.sum(q * q, dim=1, keepdim=True))
        return out + torch.stack(quad_terms, dim=0).sum(dim=0)


class NormICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, norm_dim: int, num_norm_blocks: int = 1):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth)
        self.num_norm_blocks = num_norm_blocks
        self.A = nn.Parameter(torch.randn(num_norm_blocks, norm_dim, input_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(num_norm_blocks, norm_dim))
        self.lam_raw = nn.Parameter(torch.zeros(num_norm_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        lam = torch.nn.functional.softplus(self.lam_raw) + 1e-8
        norm_terms = []
        for g in range(self.num_norm_blocks):
            u = x @ self.A[g].T + self.d[g]
            norm_terms.append(lam[g] * torch.norm(u, p=2, dim=1, keepdim=True))
        return out + torch.stack(norm_terms, dim=0).sum(dim=0)


class SOCICNN(nn.Module):
    """
    Anchor structured model:
        2-layer ReLU backbone + one Quad block + one Norm block
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        quad_rank: int,
        norm_dim: int,
        num_quad_blocks: int = 1,
        num_norm_blocks: int = 1,
    ):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth)

        self.num_quad_blocks = num_quad_blocks
        self.Ls = nn.Parameter(torch.randn(num_quad_blocks, input_dim, quad_rank) * 0.01)
        self.alpha_raw = nn.Parameter(torch.zeros(num_quad_blocks))

        self.num_norm_blocks = num_norm_blocks
        self.A = nn.Parameter(torch.randn(num_norm_blocks, norm_dim, input_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(num_norm_blocks, norm_dim))
        self.lam_raw = nn.Parameter(torch.zeros(num_norm_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)

        alpha = torch.nn.functional.softplus(self.alpha_raw) + 1e-8
        quad_terms = []
        for h in range(self.num_quad_blocks):
            q = x @ self.Ls[h]
            quad_terms.append(0.5 * alpha[h] * torch.sum(q * q, dim=1, keepdim=True))
        out_quad = torch.stack(quad_terms, dim=0).sum(dim=0)

        lam = torch.nn.functional.softplus(self.lam_raw) + 1e-8
        norm_terms = []
        for g in range(self.num_norm_blocks):
            u = x @ self.A[g].T + self.d[g]
            norm_terms.append(lam[g] * torch.norm(u, p=2, dim=1, keepdim=True))
        out_norm = torch.stack(norm_terms, dim=0).sum(dim=0)

        return out + out_quad + out_norm

