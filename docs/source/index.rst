PGMax Reference Documentation
==============================

PGMax implements general `factor graphs <https://en.wikipedia.org/wiki/Factor_graph>`_ for discrete probabilistic graphical models (PGMs), and hardware-accelerated `differentiable loopy belief propagation (LBP) <https://en.wikipedia.org/wiki/Belief_propagation>`_ in `JAX <https://jax.readthedocs.io/en/latest/>`_.

- General factor graphs: PGMax supports easy specification of general factor graphs with potentially complicated topology, factor definitions, and discrete variables with a varying number of states.
- LBP in JAX: PGMax generates pure JAX functions implementing LBP for a given factor graph. The generated pure JAX functions run on modern accelerators (GPU/TPU), work with JAX transformations (e.g. ``vmap`` for processing batches of models/samples, ``grad`` for differentiating through the LBP iterative process), and can be easily used as part of a larger end-to-end differentiable system.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   installation

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation:

   contributing

.. toctree::
   :caption: API Documentation
   :maxdepth: 3

   modules

Indices and tables
~~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
