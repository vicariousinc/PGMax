[![continuous-integration](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml/badge.svg)](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/pgmax.svg)](https://badge.fury.io/py/pgmax)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/vicariousinc/PGMax/master.svg)](https://results.pre-commit.ci/latest/github/vicariousinc/PGMax/master)
[![codecov](https://codecov.io/gh/vicariousinc/PGMax/branch/master/graph/badge.svg?token=FrRlTDCFjk)](https://codecov.io/gh/vicariousinc/PGMax)
[![Documentation Status](https://readthedocs.org/projects/pgmax/badge/?version=latest)](https://pgmax.readthedocs.io/en/latest/?badge=latest)

# PGMax

PGMax implements general factor graphs for probabilistic graphical models (PGMs) with discrete variables, and hardware-accelerated differentiable loopy belief propagation (LBP) in [JAX](https://jax.readthedocs.io/en/latest/).

- **General factor graphs**: PGMax goes beyond pairwise PGMs, and supports arbitrary factor graph topology, including higher-order factors.
- **LBP in JAX**: PGMax generates pure JAX functions implementing LBP for a given factor graph. The generated pure JAX functions run on modern accelerators (GPU/TPU), work with JAX transformations (e.g. `vmap` for processing batches of models/samples, `grad` for differentiating through the LBP iterative process), and can be easily used as part of a larger end-to-end differentiable system.

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
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
cd PGMax
poetry shell
poetry install
pre-commit install
```

### Install on GPU

By default the above commands install JAX for CPU. If you have access to a GPU, follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda) to install JAX for GPU.

## Citing PGMax

To cite this repository
```
@software{pgmax2021github,
  author = {Guangyao Zhou* and Nishanth Kumar* and Miguel L\’{a}zaro-Gredilla and Dileep George},
  title = {{PGMax}: {F}actor graph on discrete variables and hardware-accelerated differentiable loopy belief propagation in {JAX}},
  howpublished={\url{http://github.com/vicariousinc/PGMax}},
  version = {0.2.1},
  year = {2021},
}
```
where * indicates equal contribution.
