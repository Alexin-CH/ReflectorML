import os
import torch

import matplotlib.pyplot as plt

from sources import sample_beam, coords_to_density
from plot_results import plot_results

def validate_surface(mirror_model, raytracer, step, device):
    source_coords = sample_beam(50*1000).to(device)

    deformation = mirror_model(source_coords)
    predicted_coords, _ = raytracer(source_coords, deformation)
    
    source_density = coords_to_density(source_coords)
    predicted_density = coords_to_density(predicted_coords, flip=False)
    
    data = (
        source_coords.detach().cpu(),
        source_density.detach().cpu(),
        predicted_coords.detach().cpu(),
        predicted_density.detach().cpu(),
        step
    )

    return data