import os
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sources import gray_image_to_density, density_to_coords, coords_to_density
from plot_results import plot_results

def validate_surface(mirror_model, raytracer, source_img, target_img, \
    step, batch_size, res_factor, resolution, device):

    source_density = gray_image_to_density(source_img).to(device)
    source_coords = density_to_coords(
        density_map=source_density,
        num_points=20*1000
    ).to(device).requires_grad_(True)

    gradfield = mirror_model(source_coords)
    predicted_coords = raytracer(source_coords, gradfield, mirror_model)

    x = torch.linspace(-1.2, 1.2, 100).to(device)
    y = torch.linspace(-1.2, 1.2, 100).to(device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten grid for model input
    grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    # Compute gradient field for the grid
    grid_gradfield = mirror_model(grid_coords)
    
    # Reshape back to grid for plotting/analysis
    surface_mesh = torch.flip(grid_gradfield[:, 0].view(100, 100).T, dims=[0, 1])
    
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