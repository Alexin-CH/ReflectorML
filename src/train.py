import os
import torch
import torch.nn as nn

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from geomloss import SamplesLoss
from tqdm import tqdm

from sources import coords_to_density, density_to_random_coords, \
    gray_image_to_density, sample_beam, density_square
from validation import validate_surface
from network import MirrorSurface
from raytracer import MirrorRayTracer
from monge_ampere_loss import compute_ma_losses
from plot_results import gif_from_data


# --- TRAINING LOOP ---
def train_surface(target, res_divfactor, epochs, batch_size, lr, device, gif=0):
    # Setup
    epochs = int(epochs)
    
    criterion = nn.HuberLoss()
    zero = torch.tensor(0.).to(device)

    mirror_model = MirrorSurface().to(device)
    raytracer = MirrorRayTracer(target_x=5).to(device)

    optimizer = torch.optim.AdamW(
        params=mirror_model.parameters(),
        lr=lr,
        weight_decay=1e-8
    )

    # Trying different scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=epochs // 10,
        min_lr=1e-8
    )

    # Loss Function (Optimal Transport)
    # "sinkhorn" is an approximate Wasserstein distance, fully differentiable
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=1, blur=1e-8, scaling=0.9)
    
    # Optimization Loop
    print()
    print(f"Target: {target}")
    print("Starting Optimization...")
    print()
    losses = []
    list_data = []

    # Open image
    source_img = np.array(Image.open("src/templates/circle.png"))
    source_img = torch.tensor(source_img)

    target_img = np.array(Image.open(f"src/templates/{target}.png")).mean(axis=2)
    target_img = torch.tensor(target_img)

    resolution = np.array([
        [source_img.shape[0], source_img.shape[1]],
        [target_img.shape[0], target_img.shape[1]]
    ]) // res_divfactor

    tqdm_epochs = tqdm(range(epochs+1), desc="Training")
    for step in tqdm_epochs:

        # Convert images to density map and random coordinates
        # Source
        source_density = gray_image_to_density(source_img).to(device)
        source_coords = density_to_random_coords(
            density_map=source_density,
            num_points=batch_size
        ).to(device).requires_grad_(True)

        source_density = coords_to_density(
            coords=source_coords,
            n_ubins=resolution[0, 0],
            n_vbins=resolution[0, 1]
        )
        
        # Target
        target_density = gray_image_to_density(target_img).to(device)
        target_coords = density_to_random_coords(
            density_map=target_density,
            max_size=1,
            num_points=batch_size
        ).to(device)

        target_density = coords_to_density(
            coords=target_coords,
            n_ubins=resolution[1, 0],
            n_vbins=resolution[1, 1]
        )

        # Plot figures
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(source_coords.detach().cpu()[:,0], source_coords.detach().cpu()[:,1])
        # plt.axis('equal')
        # plt.show()
        # plt.figure()
        # plt.imshow(source_density.cpu())
        # plt.show()
        # plt.figure()
        # plt.scatter(target_coords.cpu()[:,0], target_coords.cpu()[:,1])
        # plt.axis('equal')
        # plt.show()
        # plt.figure()
        # plt.imshow(target_density.cpu())
        # plt.show()
        
        # Forward Raytracing
        deformation = mirror_model(source_coords)
        predicted_coords, _ = raytracer(source_coords, deformation)
        
        # Transport Loss (Sinkhorn) - Gives global structure
        transport_loss = sinkhorn_loss(predicted_coords, target_coords)

        # Monge-Ampère Loss (PDE) - Enforces local smoothness and density
        # Also enforces strict convexity
        ma_loss, cv_loss = compute_ma_losses(
            model=mirror_model,
            source_coords=source_coords,
            source_density=source_density,
            target_coords=target_coords,
            target_density=target_density,
            resolution=resolution
        )
        
        def closure():
            alpha = 0.8
            beta = 0.8

            optimizer.zero_grad()
            physics_loss = beta * cv_loss + (1 - beta) * ma_loss

            total_loss = alpha * transport_loss + (1 - beta) * physics_loss
            loss = criterion(total_loss * 1e3, zero)
            
            loss.backward()
            return loss

        loss = closure()
        optimizer.step()
        scheduler.step(loss.item())

        losses.append((
            loss.cpu().item(),
            transport_loss.cpu().item(),
            ma_loss.cpu().item()
        ))

        tqdm_epochs.set_description(f"LR {scheduler.get_last_lr()[0]:.2e} - Loss = {loss.item():.6f}")

        if gif > 0 and step % (epochs // gif) == 0:
            data = validate_surface(
                mirror_model=mirror_model,
                raytracer=raytracer,
                source_img=source_img,
                target_img=target_img,
                step=step,
                resolution=resolution,
                device=device
            )
            list_data.append(data)
            
    print()
    print(losses[0])
    print(losses[-1])
    print()

    if gif > 0: gif_from_data(list_data, title=target, fps=5)

    return mirror_model, raytracer, losses
