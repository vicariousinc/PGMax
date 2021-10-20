import numpy as np
import pytest

from pgmax.fg import graph, groups


def test_onevar_graph():
    v_group = groups.VariableDict(15, (0,))
    fg = graph.FactorGraph(v_group)
    evidence = graph.Evidence(factor_graph=fg, value=np.zeros(1))
    assert np.all(evidence.value == 0)
    assert fg._variable_group[0].num_states == 15
    with pytest.raises(ValueError) as verror:
        graph.FToVMessages(
            factor_graph=fg, default_mode="zeros", init_value=np.zeros(1)
        )

    assert "Should specify only" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        graph.FToVMessages(factor_graph=fg, default_mode="test")

    assert "Unsupported default message mode" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        graph.Evidence(factor_graph=fg, default_mode="zeros", value=np.zeros(1))

    assert "Should specify only" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        graph.Evidence(factor_graph=fg, default_mode="test")

    assert "Unsupported default evidence mode" in str(verror.value)
    fg.add_factor([0], np.arange(15)[:, None], name="test")
    with pytest.raises(ValueError) as verror:
        fg.add_factor([0], np.arange(15)[:, None], name="test")

    assert "A factor group with the name" in str(verror.value)
    init_msgs = fg.get_init_msgs()
    init_msgs.evidence[:] = {0: np.ones(15)}
    with pytest.raises(ValueError) as verror:
        init_msgs.ftov["test", 1]

    assert "Invalid keys" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        init_msgs.ftov["test", 0] = np.zeros(1)

    assert "Given message shape" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        init_msgs.ftov[0] = np.zeros(1)

    assert "Given belief shape" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        init_msgs.ftov[1] = np.zeros(1)

    assert "Invalid keys for setting messages" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        graph.FToVMessages(factor_graph=fg, init_value=np.zeros(1)).value

    assert "Expected messages shape" in str(verror.value)
