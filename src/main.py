import time
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from train import train_surface

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"Seed: {torch.seed()}")

if __name__ == "__main__":
    
    t0 = time.time()
    for target in ["square", "spiral", "pi", "bat", "cards", "heart", "plus", "qm", "pig"]:
        trained_model, raytracer, losses, loss_report = train_surface(
            target=target,
            res_factor=2,
            epochs=2e1,
            num_batch=1,
            batch_size=2500,
            lr=1e-5,
            device=device,
            gif=40 # Validation batch size is 20k so it may delay training, 0 to disable
        )

        # Plot losses
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3,
            ncols=1,
            height_ratios=[2, 1, 0.2]
        )

        plt.suptitle(f"Losses - Target: {target}")

        ax1.plot(losses[:, 2])
        ax1.plot(losses[:, 1])
        ax1.plot(losses[:, 0])
        ax1.legend(["MA", "Transport", "Total"])
        ax1.set_xticklabels([])
        ax1.set_yscale('log')
        ax1.grid()

        ax2.plot(losses[:, -1])
        ax2.legend(["LR"])
        ax2.set_yscale('linear')
        ax2.yaxis.set_major_formatter(LogFormatterMathtext())
        ax2.grid()

        ax3.axis('off')
        ax3.text(
            0, -0.1, loss_report, transform=ax3.transAxes,
            fontsize=10, fontfamily='monospace', va='top', ha='left'
        )

        plt.savefig(f"target_{target}_loss.png", bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close('all')

        break

    t_final = time.time() - t0
    t_h = t_final // 3600
    t_m = (t_final % 3600) // 60
    t_s = t_final % 60

    print()
    print(f"Elapsed Time: {int(t_h)} hours, {int(t_m)} minutes, {int(t_s)} seconds")
    print()