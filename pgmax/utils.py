import dataclasses

import jax


def register_pytree_node_dataclass(cls):
    def _flatten(obj):
        jax.tree_flatten(dataclasses.asdict(obj))

    def _unflatten(d, children):
        cls(**d.unflatten(children))

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls
