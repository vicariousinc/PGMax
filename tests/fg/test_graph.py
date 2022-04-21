import re
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from pgmax.factors import FAC_TO_VAR_UPDATES
from pgmax.factors import enumeration as enumeration_factor
from pgmax.fg import graph, groups
from pgmax.groups import logical
from pgmax.groups import variables as vgroup


def test_factor_graph():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    fg.add_factor_by_type(
        factor_type=enumeration_factor.EnumerationFactor,
        variables=[vg[0]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
        name="test",
    )
    with pytest.raises(
        ValueError,
        match="A factor group with the name test already exists. Please choose a different name",
    ):
        fg.add_factor(
            variables=[vg[0]],
            factor_configs=np.arange(15)[:, None],
            name="test",
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A Factor of type {enumeration_factor.EnumerationFactor} involving variables {frozenset([(0, 15)])} already exists."
        ),
    ):
        fg.add_factor(
            variables=[vg[0]],
            factor_configs=np.arange(10)[:, None],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Type {groups.FactorGroup} is not one of the supported factor types {FAC_TO_VAR_UPDATES.keys()}"
        ),
    ):
        fg.add_factor_by_type(variables=[vg[0]], factor_type=groups.FactorGroup)


def test_factor_adding():
    A = vgroup.NDVariableArray(num_states=2, shape=(10,))
    B = vgroup.NDVariableArray(num_states=2, shape=(10,))
    fg = graph.FactorGraph(variables=[A, B])

    with pytest.raises(ValueError, match="Do not add a factor group with no factors."):
        fg.add_factor_group(
            factory=logical.ORFactorGroup,
            variables_for_factors=[],
        )

    variables0 = (A[0], B[0])
    variables1 = (A[1], B[1])
    ORFactor = logical.ORFactorGroup(variables_for_factors=[variables0])
    with pytest.raises(
        ValueError, match="SingleFactorGroup should only contain one factor. Got 2"
    ):
        groups.SingleFactorGroup(
            variables_for_factors=[variables0, variables1],
            factor=ORFactor,
        )


def test_bp_state():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg0 = graph.FactorGraph(vg)
    fg0.add_factor(
        variables=[vg[0]],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    fg1 = graph.FactorGraph(vg)
    fg1.add_factor(
        variables=[vg[0]],
        factor_configs=np.arange(15)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match="log_potentials, ftov_msgs and evidence should be derived from the same fg_state",
    ):
        graph.BPState(
            log_potentials=fg0.bp_state.log_potentials,
            ftov_msgs=fg1.bp_state.ftov_msgs,
            evidence=fg1.bp_state.evidence,
        )


def test_log_potentials():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    fg.add_factor(
        variables=[vg[0]],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape (10,) for factor group test."),
    ):
        fg.bp_state.log_potentials["test"] = jnp.zeros((1, 15))

    with pytest.raises(
        ValueError,
        match=re.escape("Invalid name (0, 15) for log potentials updates."),
    ):
        fg.bp_state.log_potentials[vg[0]] = np.zeros(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials shape (10,). Got (15,)")
    ):
        graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(15))

    log_potentials = graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(10))
    assert jnp.all(log_potentials["test"] == jnp.zeros(10))


def test_ftov_msgs():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    fg.add_factor(
        variables=[vg[0]],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid names for setting messages"),
    ):
        fg.bp_state.ftov_msgs[0] = np.ones(10)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Given belief shape (10,) does not match expected shape (15,) for variable (0, 15)."
        ),
    ):
        fg.bp_state.ftov_msgs[vg[0]] = np.ones(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected messages shape (15,). Got (10,)")
    ):
        graph.FToVMessages(fg_state=fg.fg_state, value=np.zeros(10))

    ftov_msgs = graph.FToVMessages(fg_state=fg.fg_state, value=np.zeros(15))
    with pytest.raises(
        TypeError, match=re.escape("'FToVMessages' object is not subscriptable")
    ):
        ftov_msgs[(10,)]


def test_evidence():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    fg.add_factor(
        variables=[vg[0]],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError, match=re.escape("Expected evidence shape (15,). Got (10,).")
    ):
        graph.Evidence(fg_state=fg.fg_state, value=np.zeros(10))

    evidence = graph.Evidence(fg_state=fg.fg_state, value=np.zeros(15))
    assert jnp.all(evidence.value == jnp.zeros(15))


def test_bp():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    fg.add_factor(
        variables=[vg[0]],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    bp = graph.BP(fg.bp_state, temperature=0)
    bp_arrays = bp.update()
    bp_arrays = bp.update(
        bp_arrays=bp_arrays,
        ftov_msgs_updates={vg[0]: np.zeros(15)},
    )
    bp_arrays = bp.run_bp(bp_arrays, num_iters=1)
    bp_arrays = replace(bp_arrays, log_potentials=jnp.zeros((10)))
    bp_state = bp.to_bp_state(bp_arrays)
    assert bp_state.fg_state == fg.fg_state
