Installation Guide
===================

If all you want is to use PGMax's functionality in your own project, go ahead and follow the `User Installation Instructions`_ below. If, however, you're interested in contributing to the development of PGMax, then go ahead and follow the `Developer Installation Instructions`_. Regardless of which install you choose, do follow the `GPU Installation`_ instructions if your system is equipped with a GPU.

User Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
From PyPI

::

    pip install pgmax

From GitHub

::

    pip install git+https://github.com/vicariousinc/PGMax.git


Developer Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    git clone https://github.com/vicariousinc/PGMax.git
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
    cd PGMax
    poetry shell
    poetry install
    pre-commit install


GPU Installation
~~~~~~~~~~~~~~~~
By default the above commands install JAX for CPU. If you have access to a GPU, follow the official instructions `here <https://github.com/google/jax#pip-installation-gpu-cuda>`_ to install JAX for GPU.