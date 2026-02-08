import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def coords_to_density(coords, nbins=200):
    y_targets = coords[:, 0]
    z_targets = coords[:, 1]

    # 2D histogram for density
    y_min, y_max = y_targets.min(), y_targets.max()
    z_min, z_max = z_targets.min(), z_targets.max()
    # expand bounds a little
    pad = 0.05 * max(abs(y_max - y_min), abs(z_max - z_min), 1.0)
    y_edges = torch.linspace(y_min - pad, y_max + pad, nbins + 1)
    z_edges = torch.linspace(z_min - pad, z_max + pad, nbins + 1)
    
    # Compute 2D histogram
    H = torch.zeros(nbins, nbins, dtype=torch.long)
    y_bin_indices = torch.bucketize(y_targets, y_edges) - 1
    z_bin_indices = torch.bucketize(z_targets, z_edges) - 1
    
    # Filter out points outside the histogram range
    valid_mask = (y_bin_indices >= 0) & (y_bin_indices < nbins) & \
                 (z_bin_indices >= 0) & (z_bin_indices < nbins)
    
    # Accumulate counts
    for y, z in zip(y_bin_indices[valid_mask], z_bin_indices[valid_mask]):
        H[z, y] += 1
    
    return H.T

def density_to_random_coords(density_map, num_points=1000):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Normalize density
    density_normalized = density_map / density_map.sum()
    flat_density = density_normalized.flatten()
    cumulative_density = torch.cumsum(flat_density, dim=0)
    
    # Generate random values
    random_values = torch.rand(num_points)
    indices = torch.searchsorted(cumulative_density, random_values)
    
    # Convert flat indices to 2D indices
    y_indices, z_indices = torch.unravel_index(indices, density_map.shape)
    
    # Add random offset within the bin
    y_coords = y_indices + torch.rand(num_points)
    z_coords = z_indices + torch.rand(num_points)
    
    return torch.column_stack((z_coords, -y_coords))

def gray_image_to_density(gray_image):
    if gray_image.max() > 1:
        gray_image = gray_image / 255.0
    return  1 - gray_image




# Example usage
def demonstrate_image_to_density():
    # Create a sample grayscale image (you can replace this with your own image)
    img = np.array(Image.open("tux.png")).mean(axis=2)
    img = torch.tensor(img)
    
    # Convert image to density map
    density_map = gray_image_to_density(img)
    
    # Generate random coordinates based on the density map
    random_coords = density_to_random_coords(density_map, num_points=250*1000)
    
    # Visualization
    plt.figure(figsize=(15,5))
    
    # Original density map
    plt.subplot(131)
    plt.title('Density Map')
    plt.imshow(density_map, cmap='viridis')
    plt.colorbar()
    
    # Original image
    plt.subplot(132)
    plt.title('Original Grayscale Image')
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    
    # Generated coordinates
    plt.subplot(133)
    plt.title('Generated Coordinates')
    plt.scatter(random_coords[:, 0], random_coords[:, 1], alpha=0.1, s=1)
    
    plt.tight_layout()
    plt.axis('equal')
    plt.show()
    
    return density_map, random_coords

# Run the demonstration
density_map, random_coords = demonstrate_image_to_density()
