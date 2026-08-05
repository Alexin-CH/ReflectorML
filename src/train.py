import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from geomloss import SamplesLoss
from tqdm import tqdm

from sources import coords_to_density, density_to_coords, \
    gray_image_to_density, density_contour_coords, reflect_frame
from validation import validate_surface
from network import MirrorSurface
from raytracer import MirrorRayTracer
from monge_ampere_loss import compute_ma_losses, integrate_map
from plot_results import gif_from_data
from annealing import anneal_weights, anneal_blur_sigma


def pct_change(a, b):
    a_f = float(a)
    b_f = float(b)
    return (b_f - a_f) / a_f * 100.0 if a_f != 0 else float('nan')


# --- TRAINING LOOP ---
def train_surface(target, epochs, N, lr, device, loss_weights,
                  adam_fraction, lbfgs_lr, lbfgs_max_iter, lbfgs_history_size,
                  gif=0, anneal=False, anneal_alpha=0.9,
                  anneal_freq=5, blur_sigma=0.0, blur_final=0.0):
    # Setup
    epochs = int(epochs)
    gif = min(max(0, gif), epochs)
    switch_epoch = int(epochs * adam_fraction)

    # Unpack sample counts: N = [N_ma, N_bc, N_data]
    N_ma, N_bc, N_data = N

    # Unpack loss weights: loss_weights = [w_ma, w_bc, w_cv, w_data]
    w_ma, w_bc, w_cv, w_data = loss_weights

    mirror_model = MirrorSurface().to(device)
    raytracer = MirrorRayTracer(target_x=10).to(device)

    # Stage 1: AdamW optimizer
    optimizer = torch.optim.AdamW(
        params=mirror_model.parameters(),
        lr=lr,
        weight_decay=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=max(1, epochs // 10),
        min_lr=lr * 1e-5
    )
    stage = "Adam"

    # Loss Function (Optimal Transport)
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=1, blur=1e-8, scaling=0.9)
    
    # Optimization Loop
    print()
    print(f"Target: {target}")
    print("Starting Optimization...")
    print()
    losses = []
    list_data = []
    weights_log = []

    # Open image
    source_img = np.array(Image.open("templates/circle.png"))
    source_img = torch.tensor(source_img, dtype=torch.long)
    source_density = gray_image_to_density(source_img).to(device)

    target_img_pil = Image.open(f"templates/{target}.png")
    target_img = torch.tensor(np.array(target_img_pil).mean(axis=2), dtype=torch.long)
    target_density = gray_image_to_density(target_img).to(device)

    # Inner data points (MA)
    source_coords_ma = density_to_coords(
        density_map=source_density,
        num_points=N_ma
    ).to(device).requires_grad_(True)

    # Contour coords (transport boundary condition)
    source_contour_coords = density_contour_coords(
        density_map=source_density,
        max_size=1,
        num_points=N_bc
    ).to(device).requires_grad_(True)

    # Target coords (transport boundary condition)
    target_contour_coords = density_contour_coords(
        density_map=target_density,
        max_size=1,
        num_points=N_bc
    ).to(device)
    target_contour_coords = reflect_frame(target_contour_coords)

    # Inner data points (OT with sinkhorn loss)
    source_coords = density_to_coords(
        density_map=source_density,
        num_points=N_data
    ).to(device).requires_grad_(True)   

    # Target coords (OT with sinkhorn loss)
    target_coords = density_to_coords(
        density_map=target_density,
        max_size=1,
        num_points=N_data
    ).to(device)
    target_coords = reflect_frame(target_coords)

    tqdm_epochs = tqdm(range(epochs+1), desc="Training", dynamic_ncols=True)
    for step in tqdm_epochs:

        # Blur width anneals toward blur_final: early steps see smooth f/g
        # (well-conditioned MA), the true sharp densities recovered at the end.
        sigma = anneal_blur_sigma(step, epochs, blur_sigma)

        # Switch from Adam to L-BFGS
        if step == switch_epoch and stage == "Adam":
            optimizer = torch.optim.LBFGS(
                params=mirror_model.parameters(),
                lr=lbfgs_lr,
                max_iter=lbfgs_max_iter,
                history_size=lbfgs_history_size,
                line_search_fn="strong_wolfe"
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=0.1,
                patience=1,
                min_lr=lbfgs_lr * 1e-5
            )
            stage = "L-BFGS"

        # Validation
        if gif > 0 and step % (epochs // gif) == 0:

            data = validate_surface(
                mirror_model=mirror_model,
                raytracer=raytracer,
                source_img=source_img,
                target_img=target_img,
                step=step,
                device=device
            )
            list_data.append(data)

        if stage == "L-BFGS":
            def closure(
                optimizer=optimizer,
                source_coords=source_coords,
                source_coords_ma=source_coords_ma,
                target_coords=target_coords,
                source_contour_coords=source_contour_coords,
                target_contour_coords=target_contour_coords,
                sigma=sigma,
                loss_weights=loss_weights
            ):
                # Unpack loss weights: loss_weights = [w_ma, w_bc, w_cv, w_data]
                w_ma, w_bc, w_cv, w_data = loss_weights

                optimizer.zero_grad()

                predicted_coords = integrate_map(mirror_model, source_coords)

                ma_loss, cv_loss = compute_ma_losses(
                    model=mirror_model,
                    source_coords=source_coords_ma,
                    source_density=source_density,
                    target_density=target_density,
                    blur_sigma=sigma
                )

                bc_predicted = integrate_map(mirror_model, source_contour_coords)
                bc_loss = sinkhorn_loss(bc_predicted, target_contour_coords) ** 2

                transport_loss = sinkhorn_loss(predicted_coords, target_coords) ** 2
                physics_loss = w_ma * ma_loss + w_cv * cv_loss

                loss = physics_loss + w_bc * bc_loss + w_data * transport_loss

                loss.backward()
                return loss

            optimizer.step(closure)

        optimizer.zero_grad()

        predicted_coords = integrate_map(mirror_model, source_coords)

        ma_loss, cv_loss = compute_ma_losses(
            model=mirror_model,
            source_coords=source_coords_ma,
            source_density=source_density,
            target_density=target_density,
            blur_sigma=sigma
        )

        bc_predicted = integrate_map(mirror_model, source_contour_coords)
        bc_loss = sinkhorn_loss(bc_predicted, target_contour_coords) ** 2

        transport_loss = sinkhorn_loss(predicted_coords, target_coords) ** 2

        # Adaptive loss balancing (Algorithm 1): rescale CV/BC/data weights
        # relative to the MA reference gradient. Mutates loss_weights in place;
        # terms acting as hard weights (w_cv=0 etc.) are skipped so they stay off.
        if stage == "Adam" and anneal and step > 0 and step % anneal_freq == 0:
            anneal_weights(
                mirror_model, ma_loss, cv_loss, bc_loss, transport_loss,
                loss_weights, anneal_alpha
            )

        # Weights may have changed via balancing; unpack fresh every step.
        w_ma, w_bc, w_cv, w_data = loss_weights

        physics_loss = w_ma * ma_loss + w_cv * cv_loss

        loss = physics_loss + w_bc * bc_loss + w_data * transport_loss

        if stage == "Adam":
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
        elif stage == "L-BFGS":
            scheduler.step(loss.item())

        # Unweighted sum of the raw loss terms (physical Total). Column 0 in
        # `losses` is this unweighted sum, not the annealed scalar held by
        # `loss`, so the reported/loss "Total" is the bare physics sum.
        unweighted_total = ma_loss + cv_loss + transport_loss + bc_loss

        losses.append((
            unweighted_total.cpu().item(),
            transport_loss.cpu().item(),
            ma_loss.cpu().item(),
            cv_loss.cpu().item(),
            bc_loss.cpu().item(),
            optimizer.param_groups[0]['lr']
        ))
        weights_log.append((w_ma, w_bc, w_cv, w_data))

        if step > epochs / 10 and loss // losses[0][0] > 1e2 and not anneal:
            raise RuntimeError("Unexpected loss evolution, exiting...")

        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN, exiting...")

        tqdm_epochs.set_description(f"[{stage}] LR {optimizer.param_groups[0]['lr']:.2e} - Loss = {loss.item():.6f}")

    losses = torch.tensor(losses)
    lmin = torch.tensor(1e-8).to(losses.device)
    fl, ll = torch.max(losses[0], lmin), torch.max(losses[-1], lmin)

    loss_report = (
        f"{'Metric':<12}{'Start':>12}{'End':>12}{'Change':>12}\n"
        f"{'Total':<12}{float(fl[0]):12.6f}{float(ll[0]):12.6f}{pct_change(fl[0], ll[0]):12.3f}%\n"
        f"{'Transport':<12}{float(fl[1]):12.6f}{float(ll[1]):12.6f}{pct_change(fl[1], ll[1]):12.3f}%\n"
        f"{'MA':<12}{float(fl[2]):12.6f}{float(ll[2]):12.6f}{pct_change(fl[2], ll[2]):12.3f}%\n"
        f"{'CV':<12}{float(fl[3]):12.6f}{float(ll[3]):12.6f}{pct_change(fl[3], ll[3]):12.3f}%\n"
        f"{'BC':<12}{float(fl[4]):12.6f}{float(ll[4]):12.6f}{pct_change(fl[4], ll[4]):12.3f}%"
    )

    print()
    print(loss_report)
    print()

    if gif > 0: gif_from_data(list_data, title=f"target_{target}", fps=5)

    # Final validation snapshot of the trained surface
    final_data = validate_surface(
        mirror_model=mirror_model,
        raytracer=raytracer,
        source_img=source_img,
        target_img=target_img,
        step=epochs,
        device=device
    )

    return mirror_model, raytracer, losses, loss_report, final_data, weights_log
