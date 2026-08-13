import os
import sys
import importlib.util

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NETWORKS = {
    "src":       os.path.join(REPO_ROOT, "src",       "network.py"),
    "src_icnn":  os.path.join(REPO_ROOT, "src_icnn",  "network.py"),
    "src_gf":    os.path.join(REPO_ROOT, "src_gf",    "network.py"),
}

# Each implementation's network may import same-dir modules (e.g. src_icnn
# imports icnn.py); load each module with only its own dir on sys.path so the
# per-implementation icnn.py resolves correctly.
def load_module(path, name):
    spec_dir = os.path.dirname(path)
    sys.path.insert(0, spec_dir)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(spec_dir)
    return module

# Each implementation exposes MirrorSurface, but with a different output:
#   src/src_icnn  -> scalar potential phi(x): (N, 1)
#   src_gf        -> SPD Jacobian field J(x): (N, 2, 2)
EXPECTED_SHAPE = {
    "src":      (16, 1),
    "src_icnn": (16, 1),
    "src_gf":   (16, 2, 2),
}

try_cuda = True
device = torch.device('cuda' if try_cuda and torch.cuda.is_available() else 'cpu')


def main():
    coords = torch.rand(16, 2, device=device)
    for name, path in NETWORKS.items():
        module = load_module(path, f"network_{name}")
        model = module.MirrorSurface().to(device)

        with torch.no_grad():
            out = model(coords)

        assert tuple(out.shape) == EXPECTED_SHAPE[name], \
            f"{name}: expected {EXPECTED_SHAPE[name]}, got {tuple(out.shape)}"

        if name == "src_gf":
            # J must be symmetric positive-definite (Cholesky construction)
            assert torch.allclose(out, out.transpose(1, 2), atol=1e-5), \
                f"{name}: J not symmetric"
            eig = torch.linalg.eigvalsh(out)
            assert (eig > 0).all(), f"{name}: J not positive-definite"

        print(f"OK  {name}: MirrorSurface -> {tuple(out.shape)}")


if __name__ == "__main__":
    main()
