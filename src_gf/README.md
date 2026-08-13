# src_gf/ — Gradient-field (SPD Jacobian)

Details: [https://alexin.cclaude.rocks/projects/reflectorml/#43-integration-of-a-dsp-jacobian-pinn](https://alexin.cclaude.rocks/projects/reflectorml/#43-integration-of-a-dsp-jacobian-pinn)

Drops the potential and learns the Jacobian field `J_T(x)` directly as an SPD matrix via a Cholesky parameterization; the map is recovered by integrating the field.
An extra **curl-free** penalty (`w_curl`) enforces the path independence of the integration.

- **Network** : `network.py` - `MirrorSurface` (SIREN -> Cholesky)
- **Output** : SPD Jacobian `J_T(x) = L L^T` : `(N, 2, 2)`
- **Monge–Ampère loss** : Jacobian of the map + curl-free penalty
- **Convexity** : **guaranteed** (SPD)

## Run

From the repository root:

```bash
make
python src_gf/main.py
```

Trains on the targets `square`, `spiral`, `bat`, `cards`, `heart`, `plus`,
`qm`, `pig`.

## Results

Placeholder — run `python src_gf/main.py` and fill in the results below
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
