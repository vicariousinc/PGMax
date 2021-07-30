Installation Guide
===================

If all you want is to use PGMax's functionality in your own project, go ahead and follow the `User`_ instructions below. If, however, you're interested in contributing to the development of PGMax, then go ahead and follow the `Developer`_ instructions.

User
~~~~

1. Install the library using pip via: ``pip install pgmax``
2. By default this installs JAX for CPU. If you'd like to use JAX with a
   GPU and specific CUDA version (highly recommended), follow the
   official instructions
   `here <https://github.com/google/jax#pip-installation-gpu-cuda>`__.

Developer
~~~~~~~~~

1. Clone this project's `GitHub
   Repository <https://github.com/vicariousinc/PGMax>`__
2. Install Poetry by following `these
   instructions <https://python-poetry.org/docs/master/>`__. Note: you
   may need to logout and log back in after running the install command
   for the ``poetry --version`` command to work in your shell
   environment.
3. Navigate to this project's directory and activate a poetry shell via
   the command ``poetry shell``. This creates and activates a virtual
   environment for you to use with this project.
4. Install the project's dependencies into your virtual environment with
   the command ``poetry install``. Your environment will now contain
   both developer and user dependencies!

   1. By default this installs JAX for CPU. If you'd like to use JAX
      with a GPU and specific CUDA version (highly recommended), follow
      the official instructions
      `here <https://github.com/google/jax#pip-installation-gpu-cuda>`__.

5. Do ``pre-commit install`` to initialize pre-commit hooks