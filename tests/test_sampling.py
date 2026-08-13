import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sources import density_to_coords, gray_image_to_density, coords_to_density, \
    reflect_frame, gaussian_blur, density_to_normalization_params

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')

res_factor = 2
batch_size = 2500

# === #

source_img = np.array(Image.open("templates/circle.png"))
source_img = torch.tensor(source_img)
source_density = gray_image_to_density(source_img).to(device)

source_coords = density_to_coords(
    density_map=source_density,
    num_points=batch_size
).to(device).requires_grad_(True)

# Source density
source_density = coords_to_density(
    coords=source_coords,
    n_ubins=source_img.shape[0] // res_factor,
    n_vbins=source_img.shape[1] // res_factor
)

# Plot figures
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.scatter(source_coords.detach().cpu()[:,0], source_coords.detach().cpu()[:,1], marker='+')
plt.axis('equal')

plt.subplot(122)
plt.imshow(source_density.cpu())
plt.show()

# === reflect_frame: orientation-preserving flip of the first axis === #
assert torch.allclose(
    reflect_frame(source_coords),
    torch.cat([-source_coords[:, 0:1], source_coords[:, 1:2]], dim=1)
)
print("OK  reflect_frame")

# === gaussian_blur: same shape, positive, still a probability measure === #
density_blur = gaussian_blur(source_density, sigma=4)
assert density_blur.shape == source_density.shape
assert (density_blur > 0).all()
assert torch.allclose(density_blur.sum(), torch.ones(1), atol=1e-3)
print("OK  gaussian_blur (sigma=4)")

# === density_to_normalization_params matches density_to_coords sampling === #
row_c, col_c, max_coord = density_to_normalization_params(source_density)
assert max_coord > 0
print(f"OK  density_to_normalization_params (max_coord={float(max_coord):.3f})")

#
# #
# # #
# #
#

for target in ["pi", "spiral", "square", "bat", "cards", "heart", "plus", "qm"]:

    target_img_pil = Image.open(f"templates/{target}.png")
    target_img = np.array(target_img_pil).mean(axis=2)

    resolution = np.array([
        [source_img.shape[0], source_img.shape[1]],
        [target_img.shape[0], target_img.shape[1]]
    ]) // res_factor

    target_img_pil = target_img_pil.resize((resolution[1][1], resolution[1][0]))
    target_img = torch.tensor(np.array(target_img_pil).mean(axis=2))
    target_density = gray_image_to_density(target_img).to(device)

    # Target coords (in the reflected frame, as in train.py)
    target_coords = reflect_frame(density_to_coords(
        density_map=target_density,
        max_size=1,
        num_points=batch_size
    ).to(device))

    # Target density
    target_density = coords_to_density(
        coords=target_coords,
        n_ubins=resolution[1, 0],
        n_vbins=resolution[1, 1]
    )

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.scatter(target_coords.cpu()[:,0], target_coords.cpu()[:,1], marker='+')
    plt.axis('equal')

    plt.subplot(122)
    plt.imshow(target_density.cpu())
    plt.show()
