import os
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sources import gray_image_to_density, density_to_coords, coords_to_density
from plot_results import plot_results
from monge_ampere_loss import integrate_map

def validate_surface(mirror_model, raytracer, source_img, target_img, \
    step, batch_size, res_factor, resolution, device):

    source_density = gray_image_to_density(source_img).to(device)
    source_coords = density_to_coords(
        density_map=source_density,
        num_points=20*1000
    ).to(device).requires_grad_(True)

    predicted_coords = integrate_map(mirror_model, source_coords)

    x = torch.linspace(-1.2, 1.2, 100).to(device)
    y = torch.linspace(-1.2, 1.2, 100).to(device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # Flatten grid for model input
    grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Jacobian field over the grid: det J measures local area change
    grid_deformation = torch.linalg.det(mirror_model(grid_coords))

    # Reshape back to grid for plotting/analysis
    surface_mesh = torch.flip(grid_deformation.view(100, 100).T, dims=[0, 1])

    source_density = coords_to_density(
        coords=source_coords,
        n_ubins=resolution[0, 0],
        n_vbins=resolution[0, 1]
    )
    predicted_density = coords_to_density(
        coords=predicted_coords,
        n_ubins=resolution[1, 0],
        n_vbins=resolution[1, 1],
        flip=False
    )

    data = (
        target_img.detach().cpu(),
        source_density.detach().cpu(),
        surface_mesh.detach().cpu(),
        predicted_coords.detach().cpu(),
        predicted_density.detach().cpu(),
        (step, batch_size, res_factor)
    )

    return data