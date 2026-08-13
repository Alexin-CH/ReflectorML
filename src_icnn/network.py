import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import icnn

class ReCU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class Softplus2(nn.Module):
    def forward(self, x):
        return F.softplus(x) ** 2


class MirrorSurface(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            icnn.ICNN(
                input_dim=2,
                hidden_dim=512,
                depth=4,
                activation=nn.ReLU(),
                residual_scale=1
            ),
            nn.Linear(1, 1)
        )

        with torch.no_grad():
            self.net[-1].weight.zero_()

    def forward(self, coords):
        # Neural offset (The freeform deformation)
        deformation = self.net(coords)
        return deformation

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        return self
