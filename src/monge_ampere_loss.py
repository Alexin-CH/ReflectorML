import torch
from torch.func import hessian, vmap, jacrev  # jacrev gives per-sample gradient (∇φ)
import matplotlib
matplotlib.use('Agg')

from sources import coords_to_density_indices

def compute_ma_losses(model, source_coords, source_density, target_density, resolution, eps=1e-8):
    # Compute per-sample Hessians of scalar potential
    single_hess = lambda x: hessian(model)(x)
    hessians = vmap(single_hess)(source_coords)
    hessians = hessians.squeeze(dim=1)

    # Determinant (using stable log-det)
    sign, logabsdet = torch.linalg.slogdet(hessians)
    det_hessians = sign * torch.exp(logabsdet.clamp(min=torch.log(torch.tensor(eps).to(logabsdet.device))))

    # Map source points via gradient and sample g at mapped coords
    grad_fn = jacrev(model)
    grads = vmap(grad_fn)(source_coords)

    # Convert coords to target density indices using mapped grads
    sources_indices = coords_to_density_indices(
        coords=source_coords,
        n_ubins=resolution[0, 0],
        n_vbins=resolution[0, 1]
    )

    grad_target_indices = coords_to_density_indices(
        coords=grads.squeeze(1),
        n_ubins=resolution[1, 0],
        n_vbins=resolution[1, 1]
    )

    # Fetch densities
    f = source_density[sources_indices[:, 0].tolist(), sources_indices[:, 1].tolist()]
    g = target_density[grad_target_indices[:, 0].tolist(), grad_target_indices[:, 1].tolist()]

    # Monge–Ampère residual
    ma_res = det_hessians * (g + eps) - (f + eps)
    ma_loss = ma_res.abs().mean()

    # Convexity penalty
    eigvals = torch.linalg.eigvalsh(hessians)
    cv_pen = torch.clamp(-eigvals, min=0.0)
    cv_loss = cv_pen.mean()

    return ma_loss, cv_loss
