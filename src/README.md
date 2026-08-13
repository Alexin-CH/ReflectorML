# src/ — SIREN scalar potential

Details: [https://alexin.cclaude.rocks/projects/reflectorml/#41-siren-like-pinn](https://alexin.cclaude.rocks/projects/reflectorml/#41-siren-like-pinn)

Baseline implementation. Learns a scalar potential `phi(x)` with a SIREN
(sine-activated MLP) network; the transport map is obtained by ray-tracing
through the resulting mirror surface:

$T(x) = raytracer(x, phi(x))$

- **Network** : `network.py` - `MirrorSurface` (SIREN)
- **Output** : scalar deformation `phi(x)` : `(N, 1)`
- **Monge–Ampère loss** : Jacobian of `T`
- **Convexity** : soft (CV loss penalizes negative Hessian eigenvalues)

## Run

From the repository root:

```bash
make
python src/main.py
```

Trains on the targets `square`, `spiral`, `bat`, `cards`, `heart`, `plus`, `qm`, `pig`.

## Results

Placeholder — run `python src/main.py` and fill in the results below
(plots are written to the repository root as `target_<target>_loss.png`,
`target_<target>_final.png`, and `target_<target>.gif`).

| Target | GIF | Final | Loss |
|---|---|---|---|
| square |  |  |  |
| spiral |  |  |  |
| bat |  |  |  |
| cards |  |  |  |
| heart |  |  |  |
| plus |  |  |  |
| qm |  |  |  |
| pig |  |  |  |
