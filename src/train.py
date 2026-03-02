import os
import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from geomloss import SamplesLoss
from tqdm import tqdm

from sources import coords_to_density, density_to_coords, \
    gray_image_to_density, coords_beam, density_beam
from validation import validate_surface
from network import MirrorSurface
from raytracer import MirrorRayTracer
from monge_ampere_loss import compute_ma_losses
from plot_results import gif_from_data


def pct_change(a, b):
    a_f = float(a)
    b_f = float(b)
    return (b_f - a_f) / a_f * 100.0 if a_f != 0 else float('nan')

# --- TRAINING LOOP ---
def train_surface(target, res_factor, epochs, batch_size, num_batch, lr, device, gif=0):
    # Setup
    epochs = int(epochs)
    gif = min(max(0, gif), epochs)
    num_batch = min(max(1, num_batch), epochs)
    
    criterion = nn.MSELoss()
    zero = torch.tensor(0.).to(device)

    mirror_model = MirrorSurface().to(device)
    raytracer = MirrorRayTracer(target_x=10).to(device)

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
        patience=5,
        min_lr=lr * 1e-5
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
    source_img = torch.tensor(source_img, dtype=torch.long)
    source_density = gray_image_to_density(source_img).to(device)

    target_img_pil = Image.open(f"src/templates/{target}.png")
    target_img = np.array(target_img_pil).mean(axis=2)

    resolution = np.array([
        [source_img.shape[0], source_img.shape[1]],
        [target_img.shape[0], target_img.shape[1]]
    ]) // res_factor

    target_img_pil = target_img_pil.resize((resolution[1][1], resolution[1][0]))
    target_img = torch.tensor(np.array(target_img_pil).mean(axis=2), dtype=torch.long)
    target_density = gray_image_to_density(target_img).to(device)

    #
    # #
    # # #
    # #
    #

    tqdm_epochs = tqdm(range(epochs+1), desc="Training", dynamic_ncols=True)
    for step in tqdm_epochs:

        if step % (epochs // num_batch) == 0 and step < epochs:
            # Convert images to density map and random coordinates
            # Source coords
            source_coords = density_to_coords(
                density_map=source_density,
                num_points=batch_size
            ).to(device).requires_grad_(True)
            
            # Target coords
            target_coords = density_to_coords(
                density_map=target_density,
                max_size=1,
                num_points=batch_size
            ).to(device)

            # Reset LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Validation
        if gif > 0 and step % (epochs // gif) == 0:

            data = validate_surface(
                mirror_model=mirror_model,
                raytracer=raytracer,
                source_img=source_img,
                target_img=target_img,
                step=step,
                batch_size=batch_size,
                res_factor=res_factor,
                resolution=resolution,
                device=device
            )
            list_data.append(data)

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
        predicted_coords = raytracer(source_coords, deformation)
        
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
            alpha = 10
            beta = 1
            gamma = 1

            optimizer.zero_grad()
            physics_loss = beta * ma_loss + gamma * cv_loss

            total_loss = alpha * transport_loss +  physics_loss
            loss = criterion(total_loss, zero)
            
            loss.backward()
            return loss

        loss = closure()

        optimizer.step()
        scheduler.step(loss.item())

        losses.append((
            loss.cpu().item(),
            transport_loss.cpu().item(),
            ma_loss.cpu().item(),
            cv_loss.cpu().item(),
            scheduler.get_last_lr()[0]
        ))

        if torch.isnan(loss) or loss // losses[0][0] > 1e2:
            raise RuntimeError("Unexpected loss evolution, exiting...")

        # Save best model
        # if loss.item() < loss_min if 'loss_min' in locals() else np.inf:
        #     mirror_model.save_model(f"{target}_best_weights.pt")
        #     loss_min = loss.item()
            
        tqdm_epochs.set_description(f"LR {scheduler.get_last_lr()[0]:.2e} - Loss = {loss.item():.6f}")
            
    losses = torch.tensor(losses)
    lmin = torch.tensor(1e-8).to(losses.device)
    fl, ll = torch.max(losses[0], lmin), torch.max(losses[-1], lmin)

    loss_report = (
        f"{'Metric':<12}{'Start':>12}{'End':>12}{'Change':>12}\n"
        f"{'Total':<12}{float(fl[0]):12.6f}{float(ll[0]):12.6f}{pct_change(fl[0], ll[0]):12.3f}%\n"
        f"{'Transport':<12}{float(fl[1]):12.6f}{float(ll[1]):12.6f}{pct_change(fl[1], ll[1]):12.3f}%\n"
        f"{'MA':<12}{float(fl[2]):12.6f}{float(ll[2]):12.6f}{pct_change(fl[2], ll[2]):12.3f}%\n"
        f"{'CV':<12}{float(fl[3]):12.6f}{float(ll[3]):12.6f}{pct_change(fl[3], ll[3]):12.3f}%\n"
    )

    print()
    print(loss_report)

    if gif > 0: gif_from_data(list_data, title=f"target_{target}", fps=5)

    return mirror_model, raytracer, losses, loss_report
