import os
import torch

import matplotlib.pyplot as plt

from sources import gray_image_to_density, density_to_random_coords, coords_to_density
from plot_results import plot_results

def validate_surface(mirror_model, raytracer, source_img, target_img, step, resolution, device):
    source_density = gray_image_to_density(source_img).to(device)
    source_coords = density_to_random_coords(
        density_map=source_density,
        num_points=75*1000
    ).to(device).requires_grad_(True)

    deformation = mirror_model(source_coords)
    predicted_coords, _ = raytracer(source_coords, deformation)
    
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
        predicted_coords.detach().cpu(),
        predicted_density.detach().cpu(),
        step
    )

    return data