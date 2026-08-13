# ReflectorML

Project page: [https://alexin.cclaude.rocks/projects/reflectorml/](https://alexin.cclaude.rocks/projects/reflectorml/)

## Introduction

Free-form reflector design is essential in optics for precisely shaping light distributions, with applications in automotive lighting, energy-efficient LED optics, laser-based manufacturing, aerospace systems, and medical imaging.

This problem is mathematically formulated as a non-linear Monge-Ampère equation ([Wikipedia](https://en.wikipedia.org/wiki/Monge%E2%80%93Amp%C3%A8re_equation)), which defines the mapping between a given light source and a prescribed target intensity.

### Far-field (parallel-beam) Monge–Ampère equation

$\det\big(D^2\varphi(x)\big)=\dfrac{f(x)}{g(\nabla\varphi(x))},\;x\in\Omega$

- **$\varphi$ :** convex potential $\Omega \rightarrow \mathbb{R}$
- **$D^2\varphi$ :** Hessian of $\varphi$
- **$f$ :** source density on $\Omega$  
- **$g$ :** target density at $y=\nabla\varphi(x)$

However, traditional numerical solvers for this equation are computationally expensive and often struggle with convergence, particularly in complex boundary conditions.
Developing efficient and robust methods to solve this problem is crucial for advancing high-performance optical designs in both scientific and industrial applications.

By embedding the governing equations - such as the Monge-Ampère equation - directly into the learning process, our approach ensures physically consistent solutions while significantly reducing computational costs.
Unlike purely data-driven models, PINNs do not rely solely on labeled data but instead enforce optical constraints during training, improving solution accuracy for specific problem instances.
This framework accelerates the inverse design process and provides a computationally efficient alternative to traditional numerical solvers.

## Description

This project implements a hybrid method that aims to use both:
- PyTorch raytracer (with automatic differentiation) with a transport loss
- Physical loss based on the Monge-Ampere equation.

## Implementations

This repository contains three parallel implementations of the same reflector problem:

| Directory | Approach | Network | Network output | Monge-Ampère loss |
|---|---|---|---|---|
| `src/` | SIREN scalar potential | `MirrorSurface` (SIREN) | Scalar potential `phi(x)` | Hessian of `phi` |
| `src_icnn/` | Input Convex Neural Network potential | `MirrorSurface` (ICNN) | Scalar potential `phi(x)` | Hessian of `phi` |
| `src_gf/` | Gradient-field | `MirrorSurface` (SIREN) | SPD Jacobian `J(x) = D^2 phi` | Jacobian of the map + curl-free penalty |

Run each implementation from the repository root (templates and tests are shared):

```bash
make
python src/main.py        # potential-based
python src_icnn/main.py   # ICNN potential
python src_gf/main.py     # gradient-field
```

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Required libraries (listed in `requirements.txt`)

### Installation

Clone the repository:

```bash
git clone  https://github.com/Alexin-CH/ReflectorML.git
cd ReflectorML
```

Install the required dependencies:
```bash
make
```

- - -

## Results

Each implementation documents its results in its own README:

- [src/](src/README.md) — SIREN scalar potential
- [src_icnn/](src_icnn/README.md) — ICNN scalar potential
- [src_gf/](src_gf/README.md) — gradient-field (SPD Jacobian)

- - -

## Acknowledgments

This project is inspired by several papers:
- **"A Neural Network Approach for Solving the Monge-Ampère Equation with Transport Boundary Condition"**  
    You can read the paper [here](https://doi.org/10.48550/arXiv.2410.19496).  
- **"Input Convex Neural Networks"**  
    You can read the paper [here](https://doi.org/10.48550/arXiv.1609.07152)  
- **"Convex Physics Informed Neural Networks for the Monge-Ampère Optimal Transport Problem"**  
    You can read the paper [here](https://doi.org/10.48550/arXiv.2501.10162)

