# Implementation Comparison

ReflectorML contains three parallel implementations of the same free-form
reflector problem (Monge-Ampere + optimal transport). After the unification
pass the three share the full training pipeline and differ only in the
**network parameterization**, which is the scientific variable of the project.

## How to run

All three run from the repository root with the same interface and the same
canonical hyperparameters:

```bash
make
python src/main.py        # SIREN scalar potential
python src_icnn/main.py   # ICNN scalar potential
python src_gf/main.py     # gradient-field (SPD Jacobian)
```

Canonical config (identical in all three `main.py`):

| Param | Value |
|---|---|
| `loss_weights` | `[1,1,1,1]` (src / src_icnn), `[1,1,1,1,1]` (src_gf, + curl) |
| `epochs` | 200 |
| `lr` | 1e-4 |
| `adam_fraction` | 0.95 (Adam ~ 95% of epochs, then L-BFGS) |
| `lbfgs_lr` | 1 |
| `lbfgs_history_size` / `lbfgs_max_iter` | 10 / 30 |
| `anneal` | True |
| `blur_sigma` | 10 |

## Shared pipeline (byte-identical across all three)

The following files are **identical** in `src/`, `src_icnn/`, `src_gf/`:

| File | Role |
|---|---|
| `main.py` | Entry point, loss/weight/LR plotting, GIF + final snapshot |
| `train.py` | AdamW -> L-BFGS loop, closure, annealing, logging, validation |
| `monge_ampere_loss.py` | Reflected-frame MA residual, support/det masking, CV loss, density blur |
| `sources.py` | Coordinate/density sampling, `reflect_frame`, `gaussian_blur`, normalization |
| `annealing.py` | `anneal_weights` (Wang 2001.04536 Algo 1), `anneal_blur_sigma` |
| `validation.py` | Validation snapshot + surface/density grid |
| `plot_results.py` | Result / GIF plotting |
| `raytracer.py` | Differentiable mirror raytracer |

`src/` and `src_icnn/` are byte-identical for every one of these files.

## The intentional difference: `network.py` (+ `icnn.py`)

The three folders exist to compare network parameterizations of the transport
map. Everything else in the training loop is the same so that the comparison
isolates the architecture.

| | `src/` | `src_icnn/` | `src_gf/` |
|---|---|---|---|
| Network file | `network.py` | `network.py` + `icnn.py` | `network.py` + `icnn.py` |
| `MirrorSurface` | SIREN (sine MLP) | ICNN (input-convex ReLU backbone) | SIREN -> Cholesky SPD Jacobian |
| Output | scalar `phi(x)` : `(N, 1)` | scalar `phi(x)` : `(N, 1)` | `J(x) = L L^T` : `(N, 2, 2)` |
| Convexity of map | soft (CV loss on Hessian) | **guaranteed** by construction | **guaranteed** (SPD) |
| Transport map | `raytracer(x, phi(x))` | `raytracer(x, phi(x))` | `integrate_map(model, x)` (path-integrated J) |

`src/` is the scalar-potential SIREN baseline. `src_icnn/` replaces the SIREN
with an input-convex neural network, so the potential is convex by
construction. `src_gf/` drops the potential altogether and learns the
Jacobian field directly as an SPD matrix (Cholesky parameterization); the map
is recovered by integrating the field, and an extra **curl-free** term
(`w_curl`) enforces that `J = D^2 phi` for a single scalar potential.

Consequences of the different parameterizations:

- `src_gf/train.py` calls `compute_ma_losses(model, ...)` (no raytracer) and
  uses `integrate_map` for the BC/transport terms; it also logs a 6th `curl`
  loss column and a 5th weight.
- `src_gf/validation.py` plots `det J` (local area change) instead of the
  scalar surface mesh.
- `src_icnn/network.py` (`ReCU`/`Softplus2`/`QuadICNN`/`NormICNN`/`SOCICNN`)
  and `src_gf/icnn.py` contain additional ICNN variants not wired into the
  current `train.py`; `MirrorSurface` is the active class in both.

## Loss terms (all three)

- **MA** `w_ma` : log-space Monge-Ampere residual on the map Jacobian,
  masked to the target support, with pixel->normalized coordinate Jacobian
  correction; `src`/`src_icnn` identical, `src_gf` identical plus curl.
- **CV** `w_cv` : convexity — penalize negative Hessian eigenvalues.
- **BC** `w_bc` : boundary condition — Sinkhorn OT between contour maps.
- **Transport** `w_data` : Sinkhorn OT between interior mapped / target
  samples.
- **Curl** `w_curl` (src_gf only) : `J` integrability (`dJ11/dy = dJ12/dx`,
  `dJ22/dx = dJ12/dy`).

All four (five in gf) weights start at 1 and, when `anneal=True`, are rescaled
during the Adam stage so the mean gradient magnitude of each term matches the
MA reference (Wang et al. Algorithm 1). The reported "Total" is the
**unweighted** sum of the raw terms.
