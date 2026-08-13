# src_icnn/ — ICNN scalar potential

Replaces the SIREN baseline with an input-convex neural network (ICNN), so the
potential is convex by construction and the transport map is theoretically a
monotone map. Uses `network.py` + `icnn.py`.

- **Network** : `network.py` + `icnn.py` — `MirrorSurface` (ICNN)
- **Output** : scalar `phi(x)` : `(N, 1)`
- **Monge–Ampère loss** : Hessian of `phi`
- **Convexity** : **guaranteed** by construction

## Run

From the repository root:

```bash
make
python src_icnn/main.py
```

Trains on the targets `square`, `spiral`, `bat`, `cards`, `heart`, `plus`,
`qm`, `pig`.

## Results

Placeholder — run `python src_icnn/main.py` and fill in the results below
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
