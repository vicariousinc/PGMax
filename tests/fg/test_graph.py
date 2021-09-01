import numpy as np
import pytest

from pgmax.fg import graph, groups


def test_onevar_graph():
    v_group = groups.GenericVariableGroup(15, (0,))
    fg = graph.FactorGraph(v_group)
    assert fg._variable_group[0].num_states == 15
    with pytest.raises(ValueError) as verror:
        graph.FToVMessages(
            factor_graph=fg, default_mode="zeros", init_value=np.zeros(1)
        )

    assert "Should specify only" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        graph.FToVMessages(factor_graph=fg, default_mode="test")

    assert "Unsupported default message mode" in str(verror.value)
    fg.add_factor([0], np.arange(15)[:, None], name="test")
    init_msgs = fg.get_init_msgs()
    with pytest.raises(ValueError) as verror:
        init_msgs["test", 1]

    assert "Invalid keys" in str(verror.value)
    init_msgs.default_mode = "test"
    with pytest.raises(ValueError) as verror:
        init_msgs["test", 0]

    assert "Unsupported default message mode" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        init_msgs["test", 0] = np.zeros(1)

    assert "Given message shape" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        init_msgs[0] = np.zeros(1)

    assert "Given belief shape" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        init_msgs[1] = np.zeros(1)

    assert "Invalid keys for setting messages" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        graph.FToVMessages(factor_graph=fg, init_value=np.zeros(1)).value

    assert "Expected messages shape" in str(verror.value)
