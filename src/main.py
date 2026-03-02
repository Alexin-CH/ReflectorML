import time
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from train import train_surface

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"Seed: {torch.seed()}")

if __name__ == "__main__":
    
    t0 = time.time()
    for target in ["pi", "spiral", "square", "bat", "cards", "heart", "plus", "qm"]:
        trained_model, raytracer, losses = train_surface(
            target=target,
            res_factor=2,
            epochs=2e2,
            num_batch=1,
            batch_size=2500,
            lr=1e-5,
            device=device,
            gif=40
        )

        # Plot losses
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        plt.suptitle(f"Losses - Target: {target}")

        ax1.plot(losses[:, 2])
        ax1.plot(losses[:, 1])
        ax1.plot(losses[:, 0])
        ax1.legend(["MA", "Transport", "Total"])
        ax1.set_yscale('log')
        ax1.grid()

        ax2.plot(losses[:, -1])
        ax2.legend(["LR"])
        ax2.set_yscale('linear')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}'))
        ax2.grid()

        plt.savefig(f"target_{target}_loss.png", bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close()

        break

    t_final = time.time() - t0
    t_h = t_final // 3600
    t_m = (t_final % 3600) // 60
    t_s = t_final % 60

    print()
    print(f"Elapsed Time: {int(t_h)} hours, {int(t_m)} minutes, {int(t_s)} seconds")
    print()