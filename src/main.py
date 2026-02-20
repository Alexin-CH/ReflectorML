import torch
import matplotlib.pyplot as plt

from train import train_surface

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"Seed: {torch.seed()}")

if __name__ == "__main__":

    for target in ["pi", "bat", "cards", "heart", "square", "plus", "qm", "spiral"]:
        trained_model, raytracer, losses = train_surface(
            target=target,
            res_factor=4,
            epochs=200,
            batch_size=2048,
            num_batch=1,
            lr=1e-4,
            device=device,
            gif=40
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
        plt.close()
