import time
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from train import train_surface
from plot_results import plot_results

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"Seed: {torch.seed()}")

if __name__ == "__main__":
    
    t0 = time.time()
    for target in ["square", "spiral", "square", "bat", "cards", "heart", "plus", "qm", "pig"]:
        trained_model, raytracer, losses, loss_report, final_data, weights_log = train_surface(
            target=target,
            N=[2_500, 1_500, 1_500],  # [N_ma, N_bc, N_data]
            loss_weights=[1, 1, 1, 1], # [w_ma, w_bc, w_cv, w_data]
            epochs=200,
            lr=1e-4,

            adam_fraction=0.95,
            lbfgs_lr=1,
            lbfgs_history_size=10,
            lbfgs_max_iter=30,

            anneal=True,
            blur_sigma=10,
            gif=40,
            device=device,
        )

        # Plot losses
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=4,
            ncols=1,
            height_ratios=[2, 1, 1, 0.2]
        )
        fig.subplots_adjust(right=0.85)

        plt.suptitle(f"Losses - Target: {target}")

        ax1.plot(losses[:, 2])
        ax1.plot(losses[:, 1])
        ax1.plot(losses[:, 4])
        ax1.plot(losses[:, 0])
        ax1.legend(["MA", "Transport", "BC", "Total"], loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xticklabels([])
        ax1.set_yscale('log')
        ax1.grid()

        wlog = torch.tensor(weights_log)
        ax2.plot(wlog[:, 0])
        ax2.plot(wlog[:, 1])
        ax2.plot(wlog[:, 2])
        ax2.plot(wlog[:, 3])
        ax2.legend([r"$w_{MA}$", r"$w_{BC}$", r"$w_{CV}$", r"$w_{data}$"], loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_yscale('linear')
        ax2.set_xticklabels([])
        ax2.grid()

        ax3.plot(losses[:, -1])
        ax3.legend(["LR"], loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_yscale('linear')
        ax3.yaxis.set_major_formatter(LogFormatterMathtext())
        ax3.grid()


        ax4.axis('off')
        ax4.text(
            0, -0.1, loss_report, transform=ax4.transAxes,
            fontsize=10, fontfamily='monospace', va='top', ha='left'
        )

        plt.savefig(f"target_{target}_loss.png", bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close('all')

        # Final result plot
        plot_results(*final_data)
        plt.savefig(f"target_{target}_final.png", bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close('all')

        break

    t_final = time.time() - t0
    t_h = t_final // 3600
    t_m = (t_final % 3600) // 60
    t_s = t_final % 60

    print()
    print(f"Elapsed Time: {int(t_h)} hours, {int(t_m)} minutes, {int(t_s)} seconds")
    print()
