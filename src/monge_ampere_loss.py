import torch
from torch.func import hessian, vmap

from sources import coords_to_density_indices

def compute_ma_losses(model, source_coords, source_density, target_coords, target_density, resolution):
    # MA
    single_hess = lambda x: hessian(model)(x)
    hessians = vmap(single_hess)(source_coords).squeeze()

    det_hessians = torch.linalg.det(hessians)

    sources_indices = coords_to_density_indices(
        coords=source_coords,
        n_ubins=resolution[0, 0],
        n_vbins=resolution[0, 1]
    )
    target_indices = coords_to_density_indices(
        coords=target_coords,
        n_ubins=resolution[1, 0],
        n_vbins=resolution[1, 1]
    )

    f = source_density[sources_indices[:, 0].tolist(), sources_indices[:, 1].tolist()]
    g = target_density[target_indices[:, 0].tolist(), target_indices[:, 1].tolist()]

    ma_losses = det_hessians * g - f

    # Also enforce strict convexity
    convexity = hessians[:, 0, 0] + hessians[:, 1, 1]
    cv_losses = torch.clamp(convexity, max=0)

    return ma_losses.abs().mean(), cv_losses.abs().mean()
