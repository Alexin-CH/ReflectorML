import torch
from torch.func import vmap, jacrev
import matplotlib
matplotlib.use('Agg')

from sources import coords_to_density_indices


def compute_ma_losses(model, source_coords, source_density, target_density, resolution, eps=1e-8):
    """
    Compute MA loss for gradient field model.
    Model outputs v = ∇φ (2D vector), we compute Jac(v) = D²φ.
    """
    # Compute Jacobian of the vector field (D²φ)
    single_jac = lambda x: jacrev(model)(x)
    jacobians = vmap(single_jac)(source_coords)
    # jacobians shape: (N, 2, 2)

    # Determinant of Jacobian = det(D²φ)
    sign, logabsdet = torch.linalg.slogdet(jacobians)
    det_jacs = sign * torch.exp(logabsdet.clamp(min=torch.log(torch.tensor(eps).to(logabsdet.device))))

    # The model output IS ∇φ, so we use it directly for the right side
    grads = model(source_coords)  # (N, 2)

    # Debug
    # print(f"[DEBUG] jacobians: mean={jacobians.mean().item():.6f}, std={jacobians.std().item():.6f}")
    # print(f"[DEBUG] det_jacs: mean={det_jacs.mean().item():.6f}, min={det_jacs.min().item():.6f}, max={det_jacs.max().item():.6f}")
    # print(f"[DEBUG] grads: mean={grads.mean().item():.6f}, std={grads.std().item():.6f}")
    # print(f"[DEBUG] source_coords range: x=[{source_coords[:,0].min():.3f}, {source_coords[:,0].max():.3f}], z=[{source_coords[:,1].min():.3f}, {source_coords[:,1].max():.3f}]")


    # Convert coords to target density indices using mapped grads
    sources_indices = coords_to_density_indices(
        coords=source_coords,
        n_ubins=resolution[0, 0],
        n_vbins=resolution[0, 1]
    )

    grad_target_indices = coords_to_density_indices(
        coords=grads,
        n_ubins=resolution[1, 0],
        n_vbins=resolution[1, 1]
    )

    # Fetch densities
    f = source_density[sources_indices[:, 0].tolist(), sources_indices[:, 1].tolist()]
    g = target_density[grad_target_indices[:, 0].tolist(), grad_target_indices[:, 1].tolist()]

    # print(f"[DEBUG] f: mean={f.mean().item():.6f}, g: mean={g.mean().item():.6f}")

    # Monge–Ampère residual
    ma_res = det_jacs * (g + eps) - (f + eps)
    ma_loss = ma_res ** 2

    # print(f"[DEBUG] ma_res: mean={ma_res.mean().item():.6f}, ma_loss={ma_loss.mean().item():.6f}")

    # Convexity penalty
    eigvals = torch.linalg.eigvalsh(jacobians)
    cv_pen = torch.clamp(-eigvals, min=0.0)
    cv_loss = torch.linalg.vector_norm(grads) ** 2 # cv_pen.mean()

    # print(f"[DEBUG] cv_loss: {cv_loss.item():.6f}")

    return ma_loss.mean(), cv_loss
