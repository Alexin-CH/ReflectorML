import torch

def coords_to_edges(coords, n_ubins=200, n_vbins=200):
    u_coords = coords[:, 1].clone()
    v_coords = coords[:, 0].clone()

    # 2D histogram for density
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    u_edges = torch.linspace(u_min, u_max, n_ubins).to(coords.device)
    v_edges = torch.linspace(v_min, v_max, n_vbins).to(coords.device)
    return u_edges, v_edges

def coords_to_density_indices(coords, n_ubins=200, n_vbins=200):
    u_coords = coords[:, 1].clone()
    v_coords = coords[:, 0].clone()
    u_edges, v_edges = coords_to_edges(coords, n_ubins, n_vbins)
    
    # Find bin indices using torch.bucketize
    u_indices = torch.bucketize(u_coords, u_edges)
    v_indices = torch.bucketize(v_coords, v_edges)
    
    # Filter out indices outside the valid range
    valid_mask = (u_indices >= 0) & (u_indices < n_ubins) & \
                 (v_indices >= 0) & (v_indices < n_vbins)
    
    return torch.stack([u_indices[valid_mask], v_indices[valid_mask]], dim=1)

def coords_to_density(coords, n_ubins=200, n_vbins=200, flip=True):
    indices = coords_to_density_indices(coords, n_ubins, n_vbins)

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
    
    # Center
    x_coords = x_coords - (x_coords.max() + x_coords.min()) / 2
    y_coords = y_coords - (y_coords.max() + y_coords.min()) / 2

    # Resize
    max_coord = torch.max(x_coords.max(), y_coords.max())
    x_coords = x_coords * max_size / max_coord
    y_coords = y_coords * max_size / max_coord

    return torch.column_stack((y_coords, -x_coords))

def gray_image_to_density(gray_image):
    if gray_image.max() > 1:
        gray_image = gray_image / 255.0
    return  1 - gray_image

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