import torch
import torch.nn as nn
import torch.nn.functional as F


class MirrorRayTracer(nn.Module):
    def __init__(self, target_x=10.0, n_grid_points=100):
        super().__init__()
        self.target_x = target_x
        self.n_grid_points = n_grid_points
        self.register_buffer('v_in', torch.tensor([0.0, 1.0, 0.0]))

    def compute_phi_grid(self, model, x_range=(-1.2, 1.2), z_range=(-1.2, 1.2)):
        """Compute φ on a 2D grid.

        Evaluates ∇φ on a regular grid, then reconstructs φ via
        cumulative trapezoidal integration:
          φ(x,z) = ∫₀ˣ v_x(x',0) dx' + ∫₀ᶻ v_z(x,z') dz'
        """
        device = next(model.parameters()).device
        nx = self.n_grid_points

        x_grid = torch.linspace(x_range[0], x_range[1], nx, device=device)
        z_grid = torch.linspace(z_range[0], z_range[1], nx, device=device)
        xx, zz = torch.meshgrid(x_grid, z_grid, indexing='ij')
        grid_coords = torch.stack([xx.flatten(), zz.flatten()], dim=1)  # (nx*nx, 2)

        # Evaluate ∇φ on the 2D grid
        gradfield = model(grid_coords)  # (nx*nx, 2)

        vx_grid = gradfield[:, 0].view(nx, nx)
        vz_grid = gradfield[:, 1].view(nx, nx)

        dx = (x_grid[1] - x_grid[0]).item()
        dz = (z_grid[1] - z_grid[0]).item()

        # Cumulative trapezoidal integration along x-axis
        phi = torch.zeros(nx, nx, device=device)
        phi[1:, :] = torch.cumsum((vx_grid[:-1, :] + vx_grid[1:, :]) / 2 * dx, dim=0)

        # Add cumulative trapezoidal integration along z-axis
        phi_z = torch.zeros(nx, nx, device=device)
        phi_z[:, 1:] = torch.cumsum((vz_grid[:, :-1] + vz_grid[:, 1:]) / 2 * dz, dim=1)
        phi += phi_z

        return x_grid, z_grid, phi

    def interpolate_phi(self, coords, x_grid, z_grid, phi_grid):
        """Interpolate φ from grid to arbitrary coordinates using bilinear interpolation."""
        # Normalize coords to [-1, 1] for grid_sample
        x_min, x_max = x_grid[0], x_grid[-1]
        z_min, z_max = z_grid[0], z_grid[-1]

        norm_x = 2 * (coords[:, 0] - x_min) / (x_max - x_min) - 1
        norm_z = 2 * (coords[:, 1] - z_min) / (z_max - z_min) - 1

        # grid_sample expects (N, C, H, W) input and (N, 2) grid in [-1, 1]
        grid_xy = torch.stack([norm_z, norm_x], dim=1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        phi_input = phi_grid.to(coords.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        phi_interp = F.grid_sample(
            phi_input, grid_xy, mode='bilinear', padding_mode='border', align_corners=True
        )

        return phi_interp.squeeze()  # (N,)

    def forward(self, coords, gradfield, model):
        """
        Args:
            coords: (N, 2) input coordinates (x, z)
            gradfield: (N, 2) vector field v = ∇φ
            model: network to evaluate gradfield at arbitrary points
        """
        x = coords[:, 0:1]
        z = coords[:, 1:2]

        # Compute φ on grid via 2D integration of ∇φ
        x_grid, z_grid, phi_grid = self.compute_phi_grid(model)

        # Interpolate φ to source coordinates
        phi = self.interpolate_phi(coords, x_grid, z_grid, phi_grid).unsqueeze(1)

        # Surface height
        base_shape = -x
        y_surf = base_shape + phi

        # Surface Normals directly from v = ∇φ
        ones = torch.ones(y_surf.shape).to(y_surf.device)
        n_unnorm = torch.cat([-gradfield[:, 0:1], ones, -gradfield[:, 1:2]], dim=1)
        n_vec = torch.nn.functional.normalize(n_unnorm, dim=1)

        # Reflection
        x_surf, z_surf = x, z
        v_in = self.v_in.unsqueeze(0).expand(coords.shape[0], 3)
        dot = torch.sum(v_in * n_vec, dim=1, keepdim=True)
        v_refl = v_in - 2 * dot * n_vec

        # Intersect with Target Plane
        t = (self.target_x - x_surf) / (v_refl[:, 0:1] + 1e-6)
        y_target = y_surf + t * v_refl[:, 1:2]
        z_target = z_surf + t * v_refl[:, 2:3]

        return torch.cat([y_target, z_target], dim=1)
