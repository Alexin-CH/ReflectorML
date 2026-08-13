import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from tqdm import tqdm

from sources import coords_to_density

def save_gif(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(
        fp=outfile,
        format='GIF',
        append_images=tqdm(imgs[1:], desc="Converting to GIF"),
        save_all=True,
        duration=int(1000/fps),
        loop=loop
    )

def gif_from_data(list_data, title="nn", fps=10):
    files = []
    for i in tqdm(range(len(list_data)), desc="Preparing data"):
        surface_mesh, source_coords, source_contour_coords, \
            target_coords, target_contour_coords, \
            predicted_coords, bc_outputs, infos = list_data[i]

        plot_results(
            surface_mesh=surface_mesh,
            source_coords=source_coords,
            source_contour_coords=source_contour_coords,
            target_coords=target_coords,
            target_contour_coords=target_contour_coords,
            predicted_coords=predicted_coords,
            bc_outputs=bc_outputs,
            infos=infos
        )

        step = infos
        file = f"plots/{step}.png"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close('all')

        files.append(file)
    
    save_gif(f"{title}.gif", files, fps=fps, loop=0)

#
# #
# # #
# #
#

def plot_results(surface_mesh, source_coords, source_contour_coords,
                 target_coords, target_contour_coords,
                 predicted_coords, bc_outputs, infos):
    source_coords = source_coords.detach().cpu()
    source_contour_coords = source_contour_coords.detach().cpu()
    target_coords = target_coords.detach().cpu()
    target_contour_coords = target_contour_coords.detach().cpu()
    predicted_coords = predicted_coords.detach().cpu()
    bc_outputs = bc_outputs.detach().cpu()

    step = infos

    LIM = 1.2
    RANGE = (-LIM, LIM, -LIM, LIM)

    # Create figure
    plt.figure(figsize=(10, 6), dpi=50)
    plt.suptitle(
        t=f"Training Step {step}",
        fontsize=16,
        fontweight='bold'
    )

    # Target density: B&W 2D histogram of sampled target points (data + bc), domain [-1.2, 1.2]^2
    ax = plt.subplot(2, 2, 1)
    target_pts = torch.cat([target_coords, target_contour_coords], dim=0)
    target_density = coords_to_density(
        target_pts,
        n_ubins=100,
        n_vbins=100,
        coord_range=RANGE
    )
    im = ax.imshow(target_density.numpy(), cmap='binary', extent=[-LIM, LIM, -LIM, LIM])
    ax.set_title("Target Density", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.2)
    ax.axis('equal')

    # Source density: B&W 2D histogram of source points (data + bc), domain [-1.2, 1.2]^2
    ax = plt.subplot(2, 2, 2)
    source_pts = torch.cat([source_coords, source_contour_coords], dim=0)
    source_density = coords_to_density(
        source_pts,
        n_ubins=100,
        n_vbins=100,
        coord_range=RANGE
    )
    im = ax.imshow(source_density.numpy(), cmap='binary', extent=[-LIM, LIM, -LIM, LIM])
    ax.set_title("Source Density", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.2)

    ax = plt.subplot(2, 2, 3)
    Z = surface_mesh.numpy()
    surf = ax.imshow(Z, cmap='viridis', extent=[-LIM, LIM, -LIM, LIM]) #plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_box_aspect([3, 3, 1])
    ax.set_title("Mirror Surface", fontweight='bold')
    plt.colorbar(surf, ax=ax, shrink=0.8, pad=0.2)

    # Output density: B&W 2D histogram of raytracer outputs (data + bc), domain [-1.2, 1.2]^2
    ax = plt.subplot(2, 2, 4)
    output_coords = torch.cat([predicted_coords, bc_outputs], dim=0)
    output_density = coords_to_density(
        output_coords,
        n_ubins=100,
        n_vbins=100,
        coord_range=RANGE
    )
    im = ax.imshow(output_density.numpy(), cmap='binary', extent=[-LIM, LIM, -LIM, LIM])
    ax.set_title("Output Density", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.2)

    # Adjust layout and display
    plt.tight_layout()
