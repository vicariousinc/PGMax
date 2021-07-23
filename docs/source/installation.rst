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
    #. By default this installs JAX for GPU with CUDA 11.1 or later. If you'd like to use JAX with a different CUDA version, do:
        .. code-block:: bash

            CUDA_VERSION=cuda101 # Options are cuda101, cuda102, cuda110, cuda111 depending on your CUDA version (10.1, 10.2, ...)
            pip install --upgrade jax jaxlib==0.1.67+$CUDA_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html

#. Do :code:`pre-commit install` to initialize pre-commit hooks