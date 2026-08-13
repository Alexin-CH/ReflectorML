import torch
import torch.nn.functional as F


def gaussian_blur(density, sigma, kernel_size=None):
    """Gaussian-blur a density map (same shape, renormalized to sum 1).

    sigma in pixels; sigma <= 0 returns the density unchanged. Used to smooth
    piecewise-constant source/target densities so f/g is smooth and the
    Monge-Ampere data is well-conditioned (see anneal_blur in train.py).
    """
    if sigma <= 0:
        return density

    if kernel_size is None:
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

    r = kernel_size // 2
    t = torch.linspace(-r, r, kernel_size, device=density.device, dtype=density.dtype)
    g1 = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    kernel = torch.outer(g1, g1).view(1, 1, kernel_size, kernel_size)

    blurred = F.conv2d(
        density[None, None],
        kernel,
        padding=r
    ).view(density.shape)

    if blurred.sum() > 0:
        blurred = blurred / blurred.sum()
    return blurred


def reflect_frame(coords):
    """Express 2D coords in the reflected (orientation-preserving) target frame.

    The base 45deg mirror maps (x, z) -> (y_target ~ -x, z), which reverses
    orientation (det = -1). Flipping the vertical (first) axis makes the map
    orientation-preserving, matching the reflected-frame Monge-Ampere loss.
    Applying the same flip to both predicted and target point clouds is an
    isometry, so sinkhorn distances are unchanged.
    """
    return torch.cat([-coords[:, 0:1], coords[:, 1:2]], dim=1)

def density_to_normalization_params(density_map):
    """Pixel-space centering and scaling matching density_to_coords.

    density_to_coords centers the sampled pixels and rescales so the larger
    axis spans [-max_size, max_size]. This returns the deterministic version of
    that map, derived from the support bounding box:
    (row_center, col_center, max_coord) in pixel units.
    """
    support = density_map > 0
    rows = torch.nonzero(support.any(dim=1), as_tuple=False).squeeze(1)
    cols = torch.nonzero(support.any(dim=0), as_tuple=False).squeeze(1)

    if rows.numel() == 0 or cols.numel() == 0:
        z = torch.tensor(0.0, device=density_map.device)
        o = torch.tensor(1.0, device=density_map.device)
        return z, z, o

    row_center = (rows[0] + rows[-1]) / 2
    col_center = (cols[0] + cols[-1]) / 2
    half_row = (rows[-1] - rows[0]) / 2
    half_col = (cols[-1] - cols[0]) / 2
    max_coord = torch.maximum(half_row, half_col)
    return row_center, col_center, max_coord

def coords_to_edges(coords, n_ubins=200, n_vbins=200):
    u_coords = coords[:, 1].clone()
    v_coords = coords[:, 0].clone()

    # 2D histogram for density
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    u_edges = torch.linspace(u_min, u_max, n_ubins).to(coords.device)
    v_edges = torch.linspace(v_min, v_max, n_vbins).to(coords.device)
    return u_edges, v_edges

def pixels_to_coords(x_pix, y_pix, max_size=1):
    """Map pixel coordinates to the normalized plane used by density_to_coords.

    x_pix, y_pix: float tensors of pixel positions (any jitter included).
    Returns column_stack((y, -x)) so the convention matches density_to_coords.
    """
    x = x_pix - (x_pix.max() + x_pix.min()) / 2
    y = y_pix - (y_pix.max() + y_pix.min()) / 2

    max_coord = torch.max(x.max(), y.max())
    if max_coord > 0:
        x = x * max_size / max_coord
        y = y * max_size / max_coord

    return torch.column_stack((y, -x))

def coords_to_density_indices(coords, n_ubins=200, n_vbins=200, coord_range=None):
    u_coords = coords[:, 1].clone()
    v_coords = coords[:, 0].clone()
    if coord_range is None:
        u_edges, v_edges = coords_to_edges(coords, n_ubins, n_vbins)
    else:
        u_min, u_max, v_min, v_max = coord_range
        u_edges = torch.linspace(u_min, u_max, n_ubins).to(coords.device)
        v_edges = torch.linspace(v_min, v_max, n_vbins).to(coords.device)
    
    # Find bin indices using torch.bucketize
    u_indices = torch.bucketize(u_coords, u_edges)
    v_indices = torch.bucketize(v_coords, v_edges)
    
    # Filter out indices outside the valid range
    valid_mask = (u_indices >= 0) & (u_indices < n_ubins) & \
                 (v_indices >= 0) & (v_indices < n_vbins)
    
    return torch.stack([u_indices[valid_mask], v_indices[valid_mask]], dim=1)

def coords_to_density(coords, n_ubins=200, n_vbins=200, flip=True, coord_range=None):
    indices = coords_to_density_indices(coords, n_ubins, n_vbins, coord_range)

    H = torch.zeros(n_ubins, n_vbins, dtype=torch.long, device=coords.device)
    for u, v in indices:
        H[u, v] += 1
    if flip:
        return torch.flip(H, dims=[0])
    else:
        return H
        
def density_to_coords(density_map, max_size=1, num_points=1000, p=1):
    # Normalize density
    density_normalized = density_map / density_map.sum()
    flat_density = density_normalized.flatten()
    cumulative_density = torch.cumsum(flat_density, dim=0)
    
    
    # Generate pseudo-random values
    values = torch.linspace(0, 1, num_points + 1).to(density_map.device)[1:]

    indices = torch.searchsorted(cumulative_density, values ** p)
    
    # Convert flat indices to 2D indices
    x_indices, y_indices = torch.unravel_index(indices, density_map.shape)
    
    # Add random offset within the bin
    x_coords = x_indices + torch.rand(num_points).to(density_map.device)
    y_coords = y_indices + torch.rand(num_points).to(density_map.device)
    
    return pixels_to_coords(x_coords, y_coords, max_size)

def density_contour_indices(density_map):
    """Pixel (row, col) indices on the contour of a (binary) density map.

    A pixel is on the contour if it lies inside the shape and has at least
    one outside 8-neighbor.
    """
    inside = density_map > 0.5 * density_map.max()
    h, w = inside.shape
    padded = torch.zeros(h + 2, w + 2, dtype=torch.bool, device=density_map.device)
    padded[1:-1, 1:-1] = inside

    outside_neighbor = torch.zeros_like(inside)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            outside_neighbor |= ~padded[1 + di:1 + di + h, 1 + dj:1 + dj + w]

    contour = inside & outside_neighbor
    return contour.nonzero()

def density_contour_coords(density_map, max_size=1, num_points=None):
    """Normalized coordinates of points sampled along the domain contour.

    Uses the same centering/scaling as density_to_coords so contour points
    live in the same plane as interior points.
    """
    indices = density_contour_indices(density_map)
    if indices.shape[0] == 0:
        return torch.zeros(0, 2, device=density_map.device)

    if num_points is not None and indices.shape[0] > num_points:
        indices = indices[torch.randperm(indices.shape[0])[:num_points]]

    # Pixel centers (uniform spacing along the 8-connected contour)
    x_pix = indices[:, 0].float() + 0.5
    y_pix = indices[:, 1].float() + 0.5
    return pixels_to_coords(x_pix, y_pix, max_size)

def gray_image_to_density(gray_image, normalize=True):
    if gray_image.max() > 1:
        gray_image = gray_image / 255.0
    density = 1 - gray_image
    if not normalize:
        return density
    else:
        return density / density.sum()

#
# #
# # #
# #
#

def coords_beam(n, radius=1):
    # Circular beam
    r = torch.sqrt(torch.rand(n, 1)) * radius
    theta = torch.rand(n, 1) * 2 * torch.pi
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    return torch.cat([x, z], dim=1).requires_grad_(True)
    
def coords_square(n, width=1):
    # Simple square
    points = (torch.rand(n, 2) - 0.5) * width  # Range -width/2 to width/2
    return points.requires_grad_(True)

def coords_ellipse(n, major_axis=2, minor_axis=0.5):
    # Elliptical shape
    theta = torch.rand(n, 1) * 2 * torch.pi
    x = major_axis * torch.cos(theta)
    z = minor_axis * torch.sin(theta)
    return torch.cat([x, z], dim=1).requires_grad_(True)

def coords_triangle(n, size=1):
    # Equilateral triangle
    points = torch.rand(n, 2)
    points = points * size
    points[:, 1] = points[:, 1] * (1 - points[:, 0])  # Ensures the points are inside the triangle
    return points.requires_grad_(True)

def coords_spiral(n, turns=3, radius=1.0, noise=0.02):
    # Archimedean spiral: r ~ t, t in [0, turns*2pi]
    t = torch.linspace(0, turns * 2 * torch.pi, steps=n).unsqueeze(1)
    r = (t / (turns * 2 * torch.pi)) * radius
    x = r * torch.cos(t) + torch.randn(n, 1) * noise
    y = r * torch.sin(t) + torch.randn(n, 1) * noise
    return torch.cat([x, y], dim=1).requires_grad_(True)

def coords_l_shape(n, arm_length=1.0, arm_width=0.2):
    # L shape composed of two rectangles joined at origin: vertical and horizontal arms
    # We'll sample proportionally to area of arms
    area_h = arm_length * arm_width
    area_v = arm_length * arm_width
    probs = torch.tensor([area_h, area_v], dtype=torch.float32)
    probs = probs / probs.sum()
    choices = torch.multinomial(probs.unsqueeze(0).expand(n, -1), num_samples=1).squeeze(1)
    pts = torch.empty(n, 2)
    # horizontal arm: x in [0, arm_length], y in [-arm_width/2, arm_width/2]
    mask_h = (choices == 0)
    nh = mask_h.sum().item()
    if nh > 0:
        xh = torch.rand(nh,1) * arm_length
        yh = (torch.rand(nh,1) - 0.5) * arm_width
        pts[mask_h] = torch.cat([xh, yh], dim=1)
    # vertical arm: x in [-arm_width/2, arm_width/2], y in [0, arm_length]
    mask_v = (choices == 1)
    nv = mask_v.sum().item()
    if nv > 0:
        xv = (torch.rand(nv,1) - 0.5) * arm_width
        yv = torch.rand(nv,1) * arm_length
        pts[mask_v] = torch.cat([xv, yv], dim=1)
    return pts.requires_grad_(True)

#
# #
# # #
# #
#

def density_beam(radius=1.0, n_ubins=200, n_vbins=200):
    # Create coordinate grid
    x = torch.linspace(-radius, radius, n_ubins)
    y = torch.linspace(-radius, radius, n_vbins)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate radial distance
    r = torch.sqrt(xx**2 + yy**2)
    
    # Create circular beam density
    density = (r <= radius).float()
    
    # Normalize density
    density = density / density.sum()
    return density

def density_square(width=1.0, n_ubins=200, n_vbins=200):
    # Create coordinate grid
    x = torch.linspace(-width/2, width/2, n_ubins)
    y = torch.linspace(-width/2, width/2, n_vbins)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Create square density
    density = ((torch.abs(xx) <= width/2) & (torch.abs(yy) <= width/2)).float()
    
    # Normalize density
    density = density / density.sum()
    
    return density