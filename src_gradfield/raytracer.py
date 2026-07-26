import torch
import torch.nn as nn


class MirrorRayTracer(nn.Module):
    def __init__(self, target_x=10.0):
        super().__init__()
        self.target_x = target_x
        self.register_buffer('v_in', torch.tensor([0.0, 1.0, 0.0]))

    def forward(self, coords, gradfield, phi):
        """
        Args:
            coords: (N, 2) source coordinates (x, z)
            gradfield: (N, 2) vector field v = ∇φ
            phi: (N,) scalar potential φ at each source coordinate
        """
        x = coords[:, 0:1]
        z = coords[:, 1:2]

        base_shape = -x
        y_surf = base_shape + phi.unsqueeze(1)

        ones = torch.ones_like(y_surf)
        n_unnorm = torch.cat([-gradfield[:, 0:1], ones, -gradfield[:, 1:2]], dim=1)
        n_vec = torch.nn.functional.normalize(n_unnorm, dim=1)

        x_surf, z_surf = x, z
        v_in = self.v_in.unsqueeze(0).expand(coords.shape[0], 3)
        dot = torch.sum(v_in * n_vec, dim=1, keepdim=True)
        v_refl = v_in - 2 * dot * n_vec

        t = (self.target_x - x_surf) / (v_refl[:, 0:1] + 1e-6)
        y_target = y_surf + t * v_refl[:, 1:2]
        z_target = z_surf + t * v_refl[:, 2:3]

        return torch.cat([y_target, z_target], dim=1)
