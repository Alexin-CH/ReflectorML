import torch
import matplotlib.pyplot as plt

from sources import sample_beam
from train import train_surface
from plot_results import plot_results, plot_results_scene

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    trained_model, raytracer, losses = train_surface(
        epochs=2e2,
        lr=1e-4,
        device=device,
        gif=20
    )

    # Plot losses
    # losses = torch.tensor(losses)
    # plt.figure()
    # plt.plot(losses[:, 2])
    # plt.plot(losses[:, 1])
    # plt.plot(losses[:, 0])
    # plt.legend(["MA", "Transport", "Total"])
    # plt.yscale('log')
    # plt.grid()
    # plt.pause(2)

    # Display results
    # source = sample_beam(50*1000, 1).to(device)
    # plot_results_scene(
    #     model=trained_model,
    #     raytracer=raytracer,
    #     source=source,
    #     num_rays_to_plot=200,
    # )
