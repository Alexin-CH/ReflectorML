import torch
import torch.nn.functional as F
from torch.func import vmap

from sources import density_to_normalization_params, gaussian_blur


def density_at_coords(density, coords, max_size=1, norm_params=None):
    """Bilinearly interpolate a (normalized) density at normalized coords.

    The pixel <-> normalized affine map matches density_to_coords
    (support-bounding-box centering and scaling), so the interpolation lives in
    the same coordinate space as the sampled coords. Differentiable w.r.t. the
    mapped coords, unlike a hard histogram bin lookup.
    """
    H, W = density.shape
    if norm_params is None:
        row_center, col_center, max_coord = density_to_normalization_params(density)
    else:
        row_center, col_center, max_coord = norm_params
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


def integrate_map(model, coords, n_pts=51, chunk_size=500):
    """Recover T from its Jacobian field by Simpson line integration.

    T(x) = int_0^1 J_T(s x) x ds, with the straight path s -> s x from the
    origin (T(0) = 0 in the source frame). Differentiable w.r.t. model params.
    Returns (N, 2) transport map in the same (reflected) frame as J_T.
    """
    if n_pts % 2 == 0:
        n_pts += 1
    device = coords.device
    N = coords.shape[0]
    s = torch.linspace(0, 1, n_pts, device=device)

    h = 1.0 / (n_pts - 1)
    weights = torch.ones(n_pts, device=device)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    # Evaluate J along the straight paths in chunks to bound GPU memory:
    # scaled has shape (n_pts, chunk, 2), feeding n_pts*chunk rows at a time.
    results = []
    for i in range(0, N, chunk_size):
        chunk = coords[i:i + chunk_size]
        M = chunk.shape[0]
        scaled = s[:, None, None] * chunk[None, :, :]         # (n_pts, M, 2)
        J = model(scaled.reshape(-1, 2)).reshape(n_pts, M, 2, 2)  # (n_pts,M,2,2)

        # J(s x) @ x  -> (n_pts, M, 2)
        integrand = torch.einsum('sMij,Mj->sMi', J, chunk)
        T = h / 3.0 * torch.sum(weights[:, None, None] * integrand, dim=0)
        results.append(T)

    return torch.cat(results, dim=0)


def curl_free_loss(model, coords, chunk_size=500):
    """Mean squared curl of the Jacobian field: J must be a genuine Hessian.

    Integrability of a symmetric field J = D^2 phi (equal mixed partials of phi)
    requires:
        dJ11/dy = dJ12/dx   and   dJ22/dx = dJ12/dy.
    Enforcing this guarantees T = int J dx = grad(phi) for a single scalar
    potential phi, i.e. a truly conservative (curl-free) transport map.
    """
    total = torch.tensor(0.0, device=coords.device)
    count = 0
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        jac = model(chunk)                                  # (M, 2, 2)
        j11 = jac[:, 0, 0]
        j12 = jac[:, 0, 1]
        j22 = jac[:, 1, 1]
        ones = torch.ones_like(j11)

        # grad of each entry w.r.t. the input coords -> (M, 2) = [d/dx, d/dy]
        g11 = torch.autograd.grad(j11, chunk, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        g12 = torch.autograd.grad(j12, chunk, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        g22 = torch.autograd.grad(j22, chunk, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]

        c1 = g11[:, 1] - g12[:, 0]   # dJ11/dy - dJ12/dx
        c2 = g22[:, 0] - g12[:, 1]   # dJ22/dx - dJ12/dy

        total = total + (c1 ** 2 + c2 ** 2).sum()
        count += c1.shape[0]
    return total / max(count, 1)


def compute_ma_losses(model, source_coords, source_density, target_density,
                      blur_sigma=0.0, eps=1e-8):
    # Support-bounding-box normalization must come from the TRUE (unblurred)
    # densities: blurring dilates the support and would corrupt the
    # pixel <-> normalized coordinate mapping. Only the VALUES are smoothed.
    norm_s = density_to_normalization_params(source_density)
    norm_t = density_to_normalization_params(target_density)

    # Smooth the densities so f/g is smooth (well-conditioned MA). With
    # blur_sigma=0 the densities are passed through unchanged.
    source_smooth = gaussian_blur(source_density, blur_sigma)
    target_smooth = gaussian_blur(target_density, blur_sigma)

    # Densities as probability measures (int f = int g = 1)
    f_norm = source_smooth / source_smooth.sum()

    # Target measured in the reflected (orientation-preserving) frame, matching
    # MOKA. The base 45deg plane maps (x,z) -> (y_target ~ -x, z) with det = -1;
    # flipping the target's vertical axis turns it into an orientation-preserving
    # map, so det >= 0 and the Monge-Ampere equation holds WITHOUT an absorbing
    # absolute value. The target density is mirrored consistently: coords[:,0]
    # is the column axis (pixels_to_coords flips rows, and density_at_coords maps
    # coords[:,0]->col idx), so a negation of the first output axis corresponds
    # to a column flip (dims=[1]).
    g_norm = torch.flip(target_smooth / target_smooth.sum(), dims=[1])

    # Differential MA: the network outputs J_T(x) directly (reflected frame).
    jacobians = model(source_coords)  # (N, 2, 2)

    # Stable log-determinant (positive near identity in the reflected frame)
    logabsdet, _ = torch.linalg.slogdet(jacobians)
    dets = torch.linalg.det(jacobians)

    # Recover the map by integration and sample g at the mapped coords
    mapped = integrate_map(model, source_coords)
    g = density_at_coords(g_norm, mapped, norm_params=norm_t)

    # Only enforce where the mapped density lands inside the target density
    support_mask = g > eps

    # Jacobian of the pixel -> normalized map: |det| = (max_size/max_coord)^2.
    # Both spaces use max_size=1, so |det A_t|/|det A_s| = (max_coord_s/max_coord_t)^2
    # must be folded into the residual for the equation to hold in normalized coords.
    _, _, max_coord_s = norm_s
    _, _, max_coord_t = norm_t

    log_f = torch.log(density_at_coords(f_norm, source_coords, norm_params=norm_s) + eps)
    log_g = torch.log(g + eps)
    log_ma_res = logabsdet - (log_f - log_g) \
        - 2 * (torch.log(max_coord_s) - torch.log(max_coord_t))

    # Degenerate maps (det <= 0) are non-physical (local folds): exclude them
    # from the log-residual, they are penalized by the CV term below instead.
    valid = support_mask & (dets > 0)

    ma_loss = ((log_ma_res ** 2) * valid).sum() / valid.sum().clamp(min=1)

    # Enforce det J_T -> +1 in the reflected frame (fold-avoidance).
    cv_loss = torch.clamp(0.0 - dets, min=0.0).mean()

    # Integrability: J must be a genuine Hessian (curl-free), so that
    # T = int J dx is exactly grad(phi) for a single scalar potential.
    curl_loss = curl_free_loss(model, source_coords)

    return ma_loss, cv_loss, curl_loss