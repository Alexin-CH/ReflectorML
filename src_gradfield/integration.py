import torch


def compute_potential(model, coords, n_pts=51, chunk_size=2500):
    if n_pts % 2 == 0:
        n_pts += 1

    device = coords.device
    N = coords.shape[0]
    t = torch.linspace(0, 1, n_pts, device=device)

    h = 1.0 / (n_pts - 1)
    weights = torch.ones(n_pts, device=device)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    results = []
    for i in range(0, N, chunk_size):
        chunk = coords[i:i + chunk_size]
        M = chunk.shape[0]

        scaled = t[:, None, None] * chunk[None, :, :]
        v = model(scaled.reshape(-1, 2)).reshape(n_pts, M, 2)
        integrand = torch.sum(v * chunk[None, :, :], dim=-1)

        phi = h / 3.0 * torch.sum(weights[:, None] * integrand, dim=0)
        results.append(phi)

        del scaled, v, integrand, phi
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return torch.cat(results, dim=0)
