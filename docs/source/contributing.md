# Contributing to PGMax

PGMax is an open-source library, and as such, we very much value contributions from the community!

## Contributing code using pull requests

We welcome pull requests, in particular for documentation upgrades, additional examples, or those issues marked with
[good first issue](https://github.com/vicariousinc/PGMax/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/google/jax/issues/new/choose) or
[Discussion](https://github.com/google/jax/discussions)
to seek feedback on your planned contribution.

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Follow the Developer Installation Instructions (link!) to install a fork of PGMax locally on a new branch.
1. Develop and implement your changes with your favorite code editor (we recommend [VS Code](https://code.visualstudio.com/))
1.


(pr-checklist)=

## JAX pull request checklist

As you prepare a JAX pull request, here are a few things to keep in mind:

### Single-change commits and pull requests

A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests typically comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may need to squash together
multiple commits. We ask that you do this prior to sending the PR for review if
possible. The `git rebase -i` command might be useful to this end.

### Linting and Type-checking

JAX uses [mypy](https://mypy.readthedocs.io/) and [flake8](https://flake8.pycqa.org/)
to statically test code quality; the easiest way to run these checks locally is via
the [pre-commit](https://pre-commit.com/) framework:

```bash
pip install pre-commit
pre-commit run --all
```

If your pull request touches documentation notebooks, this will also run some checks
on those (See {ref}`update-notebooks` for more details).