# ReflectorML — Context

## Project Overview

ReflectorML computes free-form reflector shapes that transform a light source distribution (e.g., circular beam) into a prescribed target intensity pattern (e.g., a pi symbol).

It uses **Physics-Informed Neural Networks (PINNs)** to solve the nonlinear **Monge-Ampere PDE**:

```
det(D^2 phi(x)) = f(x) / g(nabla phi(x))
```

The method combines a differentiable raytracer with optimal transport loss (Sinkhorn/Wasserstein) and a PDE residual loss (Hessian determinant constraint).

See [README.md](README.md).

## Architecture

Two parallel implementations exist:

| | `src/` (potential-based) | `src_gradfield/` (gradient-field) |
|---|---|---|
| Network output | Scalar potential `phi(x)` | Vector field `nabla phi(x)` |
| Raytracer | Autograd for normals | Integrates gradient to get `phi` |
| MA loss | Hessian of `phi` | Jacobian of `nabla phi` |

## Key Files

| File | Role |
|---|---|
| `src/main.py` | Entry point — iterates targets, calls training |
| `src/network.py` | SIREN/FINER-based `MirrorSurface` network (2D→scalar) |
| `src/raytracer.py` | Differentiable ray reflection via Snell's law |
| `src/sources.py` | Density/coordinate sampling from images |
| `src/monge_ampere_loss.py` | Monge-Ampere PDE loss (Hessian + det) |
| `src/train.py` | Training loop (AdamW → L-BFGS) |
| `src/validation.py` | Validation/evaluation during training |
| `src/plot_results.py` | Visualization and GIF generation |
| `src/icnn.py` | Input Convex Neural Network architectures |
| `templates/` | Input PNG images (circle, pi, spiral, bat, etc.) |
| `images/` | Output result images and GIFs |

## Coding Conventions

- **Style:** snake_case for functions/variables, PascalCase for classes
- **Docstrings:** sparse — only on key classes/functions
- **Type annotations:** minimal (mostly in `icnn.py` only)
- **Comments:** physics-oriented explanations, ASCII art section dividers (`# # #`)
- **Imports:** standard lib → third-party → local (loose PEP 8)
- **No formatter config** (no black/ruff/flake8 setup)
- **No `__init__.py`** — run scripts from within `src/` or `src_gradfield/` directly

## How to Run

```bash
make                 # creates venv/ and installs dependencies
python src/main.py   # runs training on all target shapes
```

Requires CUDA-capable GPU. The `try_cuda = True` flag controls device selection.

Note: `src/main.py` currently has a `break` statement — only processes the first target ("pi") by default.

## Testing

No formal test framework (no pytest/unittest). The `tests/` directory contains:

- `test_sampling.py` — manual visual verification of density sampling (no assertions)
- `sources.py` — duplicate of `src/sources.py` for test imports

Verification is done by inspecting generated plots/GIFs in `images/` and `plots/`.

## Dependencies

| Package | Purpose |
|---|---|
| `torch` (>=1.10.1) | Neural networks, autograd, GPU compute |
| `geomloss` | Sinkhorn/Wasserstein optimal transport loss |
| `numpy` (>=1.21.0) | Array operations |
| `matplotlib` (>=3.4.0) | Plotting and GIF creation |
| `tqdm` (>=4.61.0) | Training progress bars |
| `pillow` | Image I/O for template loading |
| `pandas` (>=1.3.0) | Imported but minimal use |
