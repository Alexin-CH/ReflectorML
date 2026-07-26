import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w0=30):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.w0 * x)


class GradientFieldNetwork(nn.Module):
    """Outputs a 2D vector field v(x) = ∇φ(x) directly."""
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            SineLayer(2, 512),
            SineLayer(512, 256),
            SineLayer(256, 256),
            nn.Linear(256, 2)  # 2D output: (dv/dx, dv/dy)
        )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # First layer
            limit = 1 / self.net[0].linear.weight.shape[1]
            self.net[0].linear.weight.uniform_(-limit, limit)

            # Hidden layers
            for layer in self.net[1:-1]:
                if hasattr(layer, 'w0'):
                    limit = np.sqrt(6 / layer.linear.weight.shape[1]) / layer.w0
                else:
                    limit = np.sqrt(6 / layer.linear.weight.shape[1])
                layer.linear.weight.uniform_(-limit, limit)

            limit = np.sqrt(6 / self.net[-1].weight.shape[1])
            self.net[-1].weight.uniform_(-limit, limit)

    def forward(self, coords):
        # Returns 2D vector field v = ∇φ
        return self.net(coords)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        return self
