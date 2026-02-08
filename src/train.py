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
    gray_image_to_density, sample_beam
from validation import validate_surface
from network import MirrorSurface
from raytracer import MirrorRayTracer
from monge_ampere_loss import compute_ma_losses
from plot_results import gif_from_data


# --- TRAINING LOOP ---
def train_surface(epochs, lr, device, gif=0):
    # Setup
    epochs = int(epochs)
    
    criterion = nn.HuberLoss()
    zero = torch.tensor(0.).to(device)

    mirror_model = MirrorSurface().to(device)
    raytracer = MirrorRayTracer(target_x=5).to(device)

    optimizer = torch.optim.Adam(
        params=mirror_model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    # Trying different scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode="min",
    #     factor=0.1,
    #     patience=100,
    #     min_lr=1e-8
    # )

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer=optimizer,
        total_iters=epochs * 1.1,
        power=2
    )

    # Loss Function (Optimal Transport)
    # "sinkhorn" is an approximate Wasserstein distance, fully differentiable
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    
    # Optimization Loop
    print()
    print("Starting Optimization...")
    print()
    losses = []
    list_data = []

    tqdm_epochs = tqdm(range(epochs+1), desc="Training")
    for step in tqdm_epochs:
        
        # Sample rays
        batch_size = 2048

        source_coords = sample_beam(batch_size).to(device)
        source_density = coords_to_density(source_coords)

        img = np.array(Image.open("src/tux.png")).mean(axis=2)
        img = torch.tensor(img)
        
        # Convert image to density map
        # target_density = density_square().to(device)

        target_density = gray_image_to_density(img).to(device)

        target_coords = density_to_random_coords(target_density, radius=1, num_points=batch_size).to(device)
        target_density = coords_to_density(target_coords)

        # Plot figure
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(target_coords.cpu()[:,0], target_coords.cpu()[:,1])
        # plt.show()
        # plt.figure()
        # plt.imshow(target_density.cpu())
        # plt.show()
        
        # Forward Raytracing
        deformation = mirror_model(source_coords)
        predicted_coords, _ = raytracer(source_coords, deformation)
        
        # Transport Loss (Sinkhorn) - Gives global structure
        transport_loss = sinkhorn_loss(predicted_coords, target_coords)

        distance_loss = criterion(predicted_coords, target_coords)

        # Monge-Ampère Loss (PDE) - Enforces local smoothness and density
        # Also enforces strict convexity
        ma_loss, cv_loss = compute_ma_losses(
            model=mirror_model,
            source_coords=source_coords,
            source_density=source_density,
            target_coords=target_coords,
            target_density=target_density
        )
        
        def closure():
            alpha = 0.7
            beta = 0.8

            optimizer.zero_grad()
            
            pi_loss = alpha * cv_loss + (1 - alpha) * ma_loss
            total_loss = beta * transport_loss + (1 - beta) * pi_loss + distance_loss * 0.1
            loss = criterion(total_loss * 1e3, zero)
            
            loss.backward()
            return loss

        loss = closure()
        optimizer.step()
        scheduler.step() # loss.item())

        losses.append((
            loss.cpu().item(),
            transport_loss.cpu().item(),
            ma_loss.cpu().item()
        ))

        tqdm_epochs.set_description(f"LR {scheduler.get_last_lr()[0]:.2e} - Loss = {loss.item():.6f}")

        if gif > 0 and step % (epochs // gif) == 0:
            data = validate_surface(mirror_model, raytracer, step, device)
            list_data.append(data)

            
    print()
    print(losses[0])
    print(losses[-1])
    print()

    gif_from_data(list_data)

    return mirror_model, raytracer, losses
