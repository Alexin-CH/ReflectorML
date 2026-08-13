import torch
import torch.nn.functional as F

from sources import density_to_normalization_params


def density_at_coords(density, coords, max_size=1):
    """Bilinearly interpolate a (normalized) density at normalized coords.

    The pixel <-> normalized affine map matches density_to_coords
    (support-bounding-box centering and scaling), so the interpolation lives in
    the same coordinate space as the sampled coords. Differentiable w.r.t. the
    mapped coords, unlike a hard histogram bin lookup.
    """
    H, W = density.shape
    row_center, col_center, max_coord = density_to_normalization_params(density)
    scale = max_size / max_coord

    # Inverse of pixels_to_coords: coords[:,0] = norm_col, coords[:,1] = -norm_row
    col_pix = coords[:, 0] / scale + col_center
    row_pix = -coords[:, 1] / scale + row_center

    # grid_sample grid coords with align_corners=True (pixel centers at +-1)
    g_col = 2 * col_pix / (W - 1) - 1
    g_row = 2 * row_pix / (H - 1) - 1
    grid = torch.stack([g_col, g_row], dim=1).view(1, 1, -1, 2)

    sampled = F.grid_sample(
        density.float().unsqueeze(0).unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return sampled.view(-1)


def compute_ma_losses(model, raytracer, source_coords, source_density, target_density, eps=1e-8):
    # Densities as probability measures (int f = int g = 1)
    f_norm = source_density / source_density.sum()

    # Target measured in the reflected (orientation-preserving) frame, matching
    # MOKA. The base 45deg plane maps (x,z) -> (y_target, z) with det = -1;
    # flipping the target's vertical axis turns it into an orientation-preserving
    # map, so det >= 0 and the Monge-Ampere equation holds WITHOUT an absorbing
    # absolute value. The target density is mirrored consistently: coords[:,0]
    # is the column axis (pixels_to_coords flips rows, and density_at_coords maps
    # coords[:,0]->col idx), so a negation of the first output axis corresponds
    # to a column flip (dims=[1]).
    g_norm = torch.flip(target_density / target_density.sum(), dims=[1])

    # Physical transport map T(x) = raytracer(x, phi(x)), expressed in the
    # reflected target frame: first output axis is flipped.
    def map_at(x):
        mapped = raytracer(x, model(x))
        return torch.cat([-mapped[:, 0:1], mapped[:, 1:2]], dim=1)

    mapped = map_at(source_coords)

    # Jacobian of the (pointwise) map: block-diagonal (2, 2) per source point
    ones = torch.ones_like(mapped)
    d_out_x = torch.autograd.grad(
        mapped[:, 0], source_coords, grad_outputs=ones[:, 0],
        create_graph=True, retain_graph=True
    )[0]
    d_out_z = torch.autograd.grad(
        mapped[:, 1], source_coords, grad_outputs=ones[:, 1],
        create_graph=True, retain_graph=True
    )[0]
    jacobians = torch.stack([d_out_x, d_out_z], dim=1)  # (N, 2, 2)

    # Stable log-determinant (positive in the reflected frame)
    logabsdet, _ = torch.linalg.slogdet(jacobians)

    g = density_at_coords(g_norm, mapped)

    # Only enforce where the mapped density lands inside the target density
    support_mask = g > eps

    # Jacobian of the pixel -> normalized map: |det| = (max_size/max_coord)^2.
    # Both spaces use max_size=1, so |det A_t|/|det A_s| = (max_coord_s/max_coord_t)^2
    # must be folded into the residual for the equation to hold in normalized coords.
    _, _, max_coord_s = density_to_normalization_params(source_density)
    _, _, max_coord_t = density_to_normalization_params(target_density)

    log_f = torch.log(density_at_coords(f_norm, source_coords) + eps)
    log_g = torch.log(g + eps)
    log_ma_res = logabsdet - (log_f - log_g) \
        - 2 * (torch.log(max_coord_s) - torch.log(max_coord_t))

    # Degenerate maps (det <= 0) are non-physical (local folds): exclude them
    # from the log-residual, they are penalized by the CV term below instead.
    dets = torch.linalg.det(jacobians)
    valid = support_mask & (dets > 0)

    ma_loss = ((log_ma_res ** 2) * valid).sum() / valid.sum().clamp(min=1)

    # Convexity: penalize negative Hessian eigenvalues only (SPD constraint).
    cv_loss = torch.clamp(-torch.linalg.eigvalsh(jacobians), min=0.0).mean()

    return ma_loss, cv_loss
