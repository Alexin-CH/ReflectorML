import torch
from torch.func import hessian, vmap

from sources import coords_to_density_indices

def compute_ma_losses(model, source_coords, source_density, target_coords, target_density):
    # MA
    single_hess = lambda x: hessian(model)(x)
    hessians = vmap(single_hess)(source_coords).squeeze()

    det_hessians = torch.linalg.det(hessians)

    sources_indices = coords_to_density_indices(source_coords)
    target_indices = coords_to_density_indices(target_coords)

    f = source_density[sources_indices[0], sources_indices[1]]
    g = target_density[target_indices[0], target_indices[1]]

    ma_losses = det_hessians * g - f

    # Also enforce strict convexity
    convexity = hessians[:, 0, 0] + hessians[:, 1, 1]
    cv_losses = torch.clamp(convexity, max=0)

    return ma_losses.abs().mean(), cv_losses.abs().mean()
