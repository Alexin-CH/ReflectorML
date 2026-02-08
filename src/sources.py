import torch

def coords_to_edges(coords, nbins=200):
    y_targets = coords[:, 0].clone()
    z_targets = coords[:, 1].clone()

    # 2D histogram for density
    y_min, y_max = y_targets.min(), y_targets.max()
    z_min, z_max = z_targets.min(), z_targets.max()

    y_edges = torch.linspace(y_min, y_max, nbins).to(coords.device)
    z_edges = torch.linspace(z_min, z_max, nbins).to(coords.device)
    return y_edges, z_edges

def coords_to_density_indices(coords, nbins=200):
    y_targets = coords[:, 0].clone()
    z_targets = coords[:, 1].clone()
    y_edges, z_edges = coords_to_edges(coords, nbins)
    
    # Find bin indices using torch.bucketize
    y_indices = torch.bucketize(y_targets, y_edges)
    z_indices = torch.bucketize(z_targets, z_edges)
    
    # Filter out indices outside the valid range
    valid_mask = (y_indices >= 0) & (y_indices < nbins) & \
                 (z_indices >= 0) & (z_indices < nbins)
    
    return y_indices[valid_mask], z_indices[valid_mask]

def coords_to_density(coords, nbins=200):
    y_indices, z_indices = coords_to_density_indices(coords, nbins)
    H = torch.zeros(nbins, nbins, dtype=torch.long, device=coords.device)

    for y, z in zip(y_indices, z_indices):
        H[z, y] += 1
    
    return torch.flip(H, dims=[0])

def density_to_random_coords(density_map, radius, num_points=1000):
    # Normalize density
    density_normalized = density_map / density_map.sum()
    flat_density = density_normalized.flatten()
    cumulative_density = torch.cumsum(flat_density, dim=0)
    
    # Generate random values
    random_values = torch.rand(num_points).to(density_map.device)
    indices = torch.searchsorted(cumulative_density, random_values)
    
    # Convert flat indices to 2D indices
    y_indices, z_indices = torch.unravel_index(indices, density_map.shape)
    
    # Add random offset within the bin
    y_coords = y_indices + torch.rand(num_points).to(density_map.device)
    z_coords = z_indices + torch.rand(num_points).to(density_map.device)
    
    # Center
    y_coords = y_coords - (y_coords.max() + y_coords.min()) / 2
    z_coords = z_coords - (z_coords.max() + z_coords.min()) / 2

    # Resize
    y_coords = y_coords * radius / y_coords.max()
    z_coords = z_coords * radius / z_coords.max()

    return torch.column_stack((z_coords, -y_coords))


def gray_image_to_density(gray_image):
    if gray_image.max() > 1:
        gray_image = gray_image / 255.0
    return  1 - gray_image

#
# #
# # #
# #
#

def sample_beam(n, radius=1):
    # Circular beam
    r = torch.sqrt(torch.rand(n, 1)) * radius
    theta = torch.rand(n, 1) * 2 * torch.pi
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    return torch.cat([x, z], dim=1).requires_grad_(True)
    
def sample_square(n, width=1):
    # Simple square
    points = (torch.rand(n, 2) - 0.5) * width  # Range -width/2 to width/2
    return points.requires_grad_(True)

def sample_ellipse(n, major_axis=2, minor_axis=0.5):
    # Elliptical shape
    theta = torch.rand(n, 1) * 2 * torch.pi
    x = major_axis * torch.cos(theta)
    z = minor_axis * torch.sin(theta)
    return torch.cat([x, z], dim=1).requires_grad_(True)

def sample_triangle(n, size=1):
    # Equilateral triangle
    points = torch.rand(n, 2)
    points = points * size
    points[:, 1] = points[:, 1] * (1 - points[:, 0])  # Ensures the points are inside the triangle
    return points.requires_grad_(True)

def sample_spiral(n, turns=3, radius=1.0, noise=0.02):
    # Archimedean spiral: r ~ t, t in [0, turns*2pi]
    t = torch.linspace(0, turns * 2 * torch.pi, steps=n).unsqueeze(1)
    r = (t / (turns * 2 * torch.pi)) * radius
    x = r * torch.cos(t) + torch.randn(n, 1) * noise
    y = r * torch.sin(t) + torch.randn(n, 1) * noise
    return torch.cat([x, y], dim=1).requires_grad_(True)

def sample_l_shape(n, arm_length=1.0, arm_width=0.2):
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

def density_beam(radius=1.0, nbins=200):
    # Create coordinate grid
    x = torch.linspace(-radius, radius, nbins)
    y = torch.linspace(-radius, radius, nbins)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate radial distance
    r = torch.sqrt(xx**2 + yy**2)
    
    # Create circular beam density
    density = (r <= radius).float()
    
    # Normalize density
    density = density / density.sum()
    return density

def density_square(width=1.0, nbins=200):
    # Create coordinate grid
    x = torch.linspace(-width/2, width/2, nbins)
    y = torch.linspace(-width/2, width/2, nbins)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Create square density
    density = ((torch.abs(xx) <= width/2) & (torch.abs(yy) <= width/2)).float()
    
    # Normalize density
    density = density / density.sum()
    
    return density