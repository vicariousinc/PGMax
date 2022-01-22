[![continuous-integration](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml/badge.svg)](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/pgmax.svg)](https://badge.fury.io/py/pgmax)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/vicariousinc/PGMax/master.svg)](https://results.pre-commit.ci/latest/github/vicariousinc/PGMax/master)
[![codecov](https://codecov.io/gh/vicariousinc/PGMax/branch/master/graph/badge.svg?token=FrRlTDCFjk)](https://codecov.io/gh/vicariousinc/PGMax)
[![Documentation Status](https://readthedocs.org/projects/pgmax/badge/?version=latest)](https://pgmax.readthedocs.io/en/latest/?badge=latest)

# PGMax

PGMax implements general factor graphs for discrete probabilistic graphical models (PGMs), and hardware-accelerated differentiable loopy belief propagation (LBP) in [JAX](https://jax.readthedocs.io/en/latest/).

- **General factor graphs**: PGMax supports easy specification of general factor graphs with potentially complicated topology, factor definitions, and discrete variables with a varying number of states.
- **LBP in JAX**: PGMax generates pure JAX functions implementing LBP for a given factor graph. The generated pure JAX functions run on modern accelerators (GPU/TPU), work with JAX transformations (e.g. `vmap` for processing batches of models/samples, `grad` for differentiating through the LBP iterative process), and can be easily used as part of a larger end-to-end differentiable system.

[**Installation**](#installation)
| [**Getting started**](#getting-started)

## Installation

### Install from PyPI
```
pip install pgmax
```

### Install latest version from GitHub
```
pip install git+https://github.com/vicariousinc/PGMax.git
```

### Developer
```
git clone https://github.com/vicariousinc/PGMax.git
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
cd PGMax
poetry shell
poetry install
pre-commit install
```

### Install on GPU

By default the above commands install JAX for CPU. If you have access to a GPU, follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda) to install JAX for GPU.

## Getting Started

Here are a few self-contained Colab notebooks to help you get started on using PGMax:

- [Tutorial on basic PGMax usage](https://colab.research.google.com/drive/1PQ9eVaOg336XzPqko-v_us3izEbjvWMW?usp=sharing)
- [Implementing max-product LBP](https://colab.research.google.com/drive/1mSffrA1WgQwgIiJQd2pLULPa5YKAOJOX?usp=sharing) for [Recursive Cortical Networks](https://www.science.org/doi/10.1126/science.aag2612)
- [End-to-end differentiable LBP for gradient-based PGM training](https://colab.research.google.com/drive/1yxDCLwhX0PVgFS7NHUcXG3ptMAY1CxMC?usp=sharing)



## Citing PGMax

To cite this repository
```
@software{pgmax2021github,
  author = {Guangyao Zhou* and Nishanth Kumar* and Miguel L\’{a}zaro-Gredilla and Dileep George},
  title = {{PGMax}: {F}actor graph on discrete variables and hardware-accelerated differentiable loopy belief propagation in {JAX}},
  howpublished={\url{http://github.com/vicariousinc/PGMax}},
  version = {0.2.2},
  year = {2022},
}
```
where * indicates equal contribution.
