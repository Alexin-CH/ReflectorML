import os
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sources import gray_image_to_density, density_to_coords, density_contour_coords, reflect_frame
from plot_results import plot_results

def validate_surface(mirror_model, raytracer, source_img, target_img, step, device):

    source_density = gray_image_to_density(source_img).to(device)
    source_coords = density_to_coords(
        density_map=source_density,
        num_points=10_000
    ).to(device).requires_grad_(True)

    deformation = mirror_model(source_coords)
    predicted_coords = reflect_frame(raytracer(source_coords, deformation))

    # Boundary condition coords (source contour) and their raytracer outputs
    source_contour_coords = density_contour_coords(
        density_map=source_density,
        max_size=1,
        num_points=2000
    ).to(device).requires_grad_(True)

    bc_outputs = reflect_frame(raytracer(source_contour_coords, mirror_model(source_contour_coords)))

    target_density = gray_image_to_density(target_img).to(device)
    target_coords = reflect_frame(density_to_coords(
        density_map=target_density,
        max_size=1,
        num_points=10_000
    ).to(device))
    target_contour_coords = reflect_frame(density_contour_coords(
        density_map=target_density,
        max_size=1,
        num_points=2000
    ).to(device))

    x = torch.linspace(-1.2, 1.2, 100).to(device)
    y = torch.linspace(-1.2, 1.2, 100).to(device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten grid for model input
    grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    # Compute surface deformation for the grid
    grid_deformation = mirror_model(grid_coords)
    
    # Reshape back to grid for plotting/analysis
    surface_mesh = torch.flip(grid_deformation.view(100, 100).T, dims=[0, 1])
    
    data = (
        surface_mesh.detach().cpu(),
        source_coords.detach().cpu(),
        source_contour_coords.detach().cpu(),
        target_coords.detach().cpu(),
        target_contour_coords.detach().cpu(),
        predicted_coords.detach().cpu(),
        bc_outputs.detach().cpu(),
        step
    )

    return data