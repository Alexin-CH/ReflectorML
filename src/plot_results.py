import os
import torch
import numpy as np
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

def gif_from_data(list_data, fps=10):
    files = []
    for i in tqdm(range(len(list_data)), desc="Preparing data"):
        img, source_density, \
            predicted_coords, predicted_density, \
                step = list_data[i]

        plot_results(
            image=img,
            source_density=source_density,
            predicted_coords=predicted_coords,
            predicted_density=predicted_density,
            step=step
        )

        file = f"plots/{step}.png"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close()

        files.append(file)
    
    save_gif("nn.gif", files, fps=fps, loop=0)

# --- VISUALIZATION OF RESULTS ---
def plot_results(image, source_density, predicted_coords, predicted_density, step=0):
    image = image.detach().cpu()
    source_density = source_density.detach().cpu()
    predicted_coords = predicted_coords.detach().cpu()
    predicted_density = predicted_density.detach().cpu()

    # Create figure
    plt.figure(figsize=(10, 6), dpi=50)
    plt.suptitle(f"Training Step {step}", fontsize=16, fontweight='bold')

    # Target image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Target", fontweight='bold')
    plt.axis('equal')

    # Source Density Heatmap
    plt.subplot(2, 2, 2)
    im1 = plt.imshow(source_density, cmap='Greys', 
                     aspect='auto', origin='lower')
    plt.title("Input Density", fontweight='bold')
    plt.colorbar(im1, label='Density', shrink=0.8)
    plt.axis('equal')

    # Predicted Coordinate Scatter Plot
    plt.subplot(2, 2, 3)
    plt.scatter(predicted_coords[:,0], predicted_coords[:,1], 
                s=3, marker="+", alpha=0.5, color='red', 
                label='Predicted Coordinates')
    plt.title("Output Coordinates", fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    # Predicted Density Heatmap
    plt.subplot(2, 2, 4)
    im2 = plt.imshow(predicted_density, cmap='Greys', 
                     aspect='auto', origin='lower')
    plt.title("Output Density", fontweight='bold')
    plt.colorbar(im2, label='Density', shrink=0.8)
    plt.axis('equal')

    # Adjust layout and display
    plt.tight_layout()
    
def plot_results_scene(model,
                       raytracer,
                       source,
                       grid_res=80,
                       num_rays_to_plot=200
                       ):

    source = source.detach().requires_grad_(True)
    n_samples = source.shape[0]
    device = source.device

    y_surf_samples = model(source).view(-1, 1)  # (N,1)
    hits = raytracer(source, y_surf_samples)[0]  # (N,2) -> (y_target, z_target)
    hits = hits.detach().cpu().numpy()

    y_surf_np = y_surf_samples.detach().cpu().numpy().flatten()
    source_np = source.detach().cpu().numpy()
    x_surf_np = source_np[:, 0].copy()
    z_surf_np = source_np[:, 1].copy()

    # 3) Prepare surface grid for plotting (mesh)
    gx = np.linspace(-1, 1, grid_res)
    gz = np.linspace(-1, 1, grid_res)
    GX, GZ = np.meshgrid(gx, gz, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GZ.ravel()], axis=1)
    grid_tensor = torch.from_numpy(grid_pts).float().to(device)
    with torch.no_grad():
        GY = model(grid_tensor).cpu().numpy().reshape(grid_res, grid_res)

    # 4) Prepare target hits and density map
    y_targets = hits[:, 0]
    z_targets = hits[:, 1]
    x_target_plane = raytracer.target_x

    # 2D histogram for density on (y,z)
    nbins = 200
    y_min, y_max = y_targets.min(), y_targets.max()
    z_min, z_max = z_targets.min(), z_targets.max()
    # expand bounds a little
    pad = 0.05 * max(abs(y_max - y_min), abs(z_max - z_min), 1.0)
    y_edges = np.linspace(y_min - pad, y_max + pad, nbins + 1)
    z_edges = np.linspace(z_min - pad, z_max + pad, nbins + 1)
    H, y_edges, z_edges = np.histogram2d(y_targets, z_targets, bins=[y_edges, z_edges])
    H = H.T  # transpose so rows = z, cols = y for imshow

    # 5) Create Figure: 3D scene + source/target 2D plots
    fig = plt.figure(figsize=(16, 8))

    # 3D Scene
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    # plot surface: X=GX, Y=GY, Z=GZ -> our system X (plot x) ; Y is "height" ; Z is depth
    # We want axes: X (x), Y (y), Z (z)
    ax3d.plot_surface(GX, GY, GZ, rstride=1, cstride=1, cmap='Purples', alpha=1, linewidth=0.01, antialiased=True)

    # plot target hits on the target plane
    ax3d.scatter(np.full_like(y_targets, x_target_plane), y_targets, z_targets,
                 c='black', s=2, alpha=0.6, label='Target Hits (y,z)')

    # plot rays (subset)
    idxs = np.linspace(0, n_samples - 1, min(num_rays_to_plot, n_samples), dtype=int)
    for i in idxs:
        xs = x_surf_np[i]
        zs = z_surf_np[i]
        ys = y_surf_np[i]
        yt = y_targets[i]
        zt = z_targets[i]

        # incoming ray: from (x, 5, z) to (x, ys, z)
        ax3d.plot([xs, xs], [5, ys], [zs, zs], color='red', alpha=0.4, linewidth=0.8)

        # reflected ray: from surface (xs, ys, zs) to target (x_target_plane, yt, zt)
        ax3d.plot([xs, x_target_plane], [ys, yt], [zs, zt], color='orange', alpha=0.6, linewidth=0.8)

    # draw the target plane as a wireframe for context
    # create a small grid on the target plane
    y_grid = np.linspace(y_min - pad / 2, y_max + pad / 2, 6)
    z_grid = np.linspace(z_min - pad / 2, z_max + pad / 2, 6)
    Yg, Zg = np.meshgrid(y_grid, z_grid)
    Xg = np.full_like(Yg, x_target_plane)
    ax3d.plot_wireframe(Xg, Yg, Zg, color='k', alpha=0.2)

    ax3d.set_xlabel('X (system X)')
    ax3d.set_ylabel('Y (system Y / light direction)')
    ax3d.set_zlabel('Z (system Z)')
    ax3d.set_title('3D Scene: Source -> Mirror Surface -> Target')
    ax3d.legend(loc='upper left')
    ax3d.axis('equal')

    # set a reasonable view and limits
    ax3d.view_init(elev=25, azim=-60)

    # Right: 2D plots (source and target density)
    ax_src = fig.add_subplot(2, 2, 3)
    ax_src.scatter(x_surf_np, z_surf_np, s=2, c='blue', alpha=0.6)
    ax_src.set_aspect('equal')
    ax_src.set_title('Input Beam (x, z)')
    ax_src.set_xlabel('x'); ax_src.set_ylabel('z')
    ax_src.axis('equal')
    ax_src.grid(True)

    ax_target = fig.add_subplot(2, 2, 4)
    im = ax_target.imshow(H, origin='lower', aspect='auto', cmap='Greys')
    ax_target.set_title('Output Density on Target Plane (y, z)')
    ax_target.set_xlabel('y'); ax_target.set_ylabel('z')
    plt.colorbar(im, ax=ax_target, fraction=0.046, pad=0.04)
    ax_target.axis('equal')

    base_plane = -GX
    sag = GY - base_plane

    ax_surf = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax_surf.plot_surface(GX, GZ, sag, cmap='plasma')
    ax_surf.set_title("Learned Mirror Deformation (Sag)")
    ax_surf.axis('equal')
    fig.colorbar(surf)

    plt.tight_layout()
    plt.show()