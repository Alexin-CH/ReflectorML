import torch
import matplotlib.pyplot as plt

from train import train_surface

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if __name__ == "__main__":

    for target in ["pi"]: # , "bat", "cards", "heart", "square", "plus", "qm", "spiral"]:
        trained_model, raytracer, losses = train_surface(
            target=target,
            res_divfactor=1,
            epochs=2e2,
            batch_size=1024*2,
            lr=1e-4,
            device=device,
            gif=20
        )

        # Plot losses
        plt.figure()
        plt.plot(losses[:, 2])
        plt.plot(losses[:, 1])
        plt.plot(losses[:, 0])
        plt.legend(["MA", "Transport", "Total"])
        plt.yscale('log')
        plt.grid()
        plt.savefig(f"{target}_loss.png", bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.pause(0.1)
