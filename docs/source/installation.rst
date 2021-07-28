Installation Guide
===================

If all you want is to use PGMax's functionality in your own project, go ahead and follow the `User Installation Instructions`_ below. If, however, you're interested in contributing to the development of PGMax, then go ahead and follow the `Developer Installation Instructions`_.

User Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(TODO)

Developer Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Install Poetry by following `these instructions <https://python-poetry.org/docs/master/>`_. Note: you may need to logout and log back in after running the install command for the `poetry --version` command to work in your shell environment.
#. Navigate to this project's directory and activate a poetry shell via the command :code:`poetry shell`. This creates and activates a virtual environment for you to use with this project.
#. Install the project's dependencies into your virtual environment with the command :code:`poetry install`. Your environment will now contain both developer and user dependencies!
    #. By default this installs JAX for CPU. If you'd like to use JAX with a GPU and specific CUDA version, follow the official instructions `here <https://github.com/google/jax#pip-installation-gpu-cuda>`_.
#. Do :code:`pre-commit install` to initialize pre-commit hooks