import dataclasses

import pytest

import pgmax.fg.nodes as nodes


def test_variable_frozen():
    v = nodes.Variable(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.num_states = 4


# TODO: Write more tests for this file
