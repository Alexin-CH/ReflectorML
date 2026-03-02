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
        img, source_density, surface_mesh, \
            predicted_coords, predicted_density, \
                infos = list_data[i]

        plot_results(
            image=img,
            source_density=source_density,
            surface_mesh=surface_mesh,
            predicted_coords=predicted_coords,
            predicted_density=predicted_density,
            infos=infos
        )

        step = infos[0]
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

def plot_results(image, source_density, surface_mesh, predicted_coords, predicted_density, infos):
    image = image.detach().cpu()
    source_density = source_density.detach().cpu()
    predicted_coords = predicted_coords.detach().cpu()
    predicted_density = predicted_density.detach().cpu()

    step, batch_size, res_factor = infos

    # Create figure
    plt.figure(figsize=(10, 6), dpi=50)
    plt.suptitle(
        t=f"Batch size: {batch_size} - Res factor: {res_factor} - Training Step {step}",
        fontsize=16,
        fontweight='bold'
    )

    # Target image
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Target Density", fontweight='bold')
    plt.axis('equal')

    # Source Density Heatmap
    plt.subplot(2, 2, 2)
    im1 = plt.imshow(source_density, cmap='binary', 
                     aspect='auto', origin='lower')
    plt.title("Input Density", fontweight='bold')
    plt.colorbar(im1, label='Density', shrink=0.8)
    plt.axis('equal')

    ax = plt.subplot(2, 2, 3)
    Z = surface_mesh.numpy()
    surf = ax.imshow(Z, cmap='viridis') #plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_box_aspect([3, 3, 1])
    ax.set_title("Mirror Surface", fontweight='bold')
    plt.colorbar(surf, ax=ax, shrink=0.8, pad=0.2)

    # Predicted Density Heatmap
    plt.subplot(2, 2, 4)
    im2 = plt.imshow(predicted_density, cmap='binary', 
                     aspect='auto', origin='lower')
    plt.title("Output Density", fontweight='bold')
    plt.colorbar(im2, label='Density', shrink=0.8)
    plt.axis('equal')

    # Adjust layout and display
    plt.tight_layout()
