import re
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgmax.factors import enumeration as enumeration_factor
from pgmax.fg import graph
from pgmax.groups import enumeration
from pgmax.groups import variables as vgroup


def test_factor_graph():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)

    with pytest.raises(
        ValueError,
        match="A Factor or a FactorGroup is required",
    ):
        fg.add_factors(factor_group=None, factor=None)

    factor = enumeration_factor.EnumerationFactor(
        variables=[vg[0]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )

    factor_group = enumeration.EnumerationFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )
    with pytest.raises(
        ValueError,
        match="Cannot simultaneously add a Factor and a FactorGroup",
    ):
        fg.add_factors(factor_group=factor_group, factor=factor)

    fg.add_factors(factor=factor)

    factor_group = enumeration.EnumerationFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A Factor of type {enumeration_factor.EnumerationFactor} involving variables {frozenset([((vg.__hash__(), 0), 15)])} already exists."
        ),
    ):
        fg.add_factors(factor_group)


def test_bp_state():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg0 = graph.FactorGraph(vg)
    factor = enumeration_factor.EnumerationFactor(
        variables=[vg[0]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )
    fg0.add_factors(factor=factor)

    fg1 = graph.FactorGraph(vg)
    fg1.add_factors(factor=factor)

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
    factor_group = enumeration.EnumerationFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.add_factors(factor_group)

    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape (10,) for factor group."),
    ):
        fg.bp_state.log_potentials[factor_group] = jnp.zeros((1, 15))

    with pytest.raises(
        ValueError,
        match=re.escape("Invalid FactorGroup for log potentials updates."),
    ):
        factor_group2 = enumeration.EnumerationFactorGroup(
            variables_for_factors=[[vg[0]]],
            factor_configs=np.arange(10)[:, None],
        )
        fg.bp_state.log_potentials[factor_group2] = jnp.zeros((1, 15))

    with pytest.raises(
        ValueError,
        match=re.escape("Invalid FactorGroup for log potentials updates."),
    ):
        fg.bp_state.log_potentials[vg[0]] = np.zeros(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials shape (10,). Got (15,)")
    ):
        graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(15))

    log_potentials = graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(10))
    assert jnp.all(log_potentials[factor_group] == jnp.zeros(10))


def test_ftov_msgs():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    factor_group = enumeration.EnumerationFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.add_factors(factor_group)

    with pytest.raises(
        ValueError,
        match=re.escape("Provided variable is not in the FactorGraph"),
    ):
        fg.bp_state.ftov_msgs[0] = np.ones(10)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Given belief shape (10,) does not match expected shape (15,) for variable (({vg.__hash__()}, 0), 15)."
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
    factor_group = enumeration.EnumerationFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.add_factors(factor_group)

    with pytest.raises(
        ValueError, match=re.escape("Expected evidence shape (15,). Got (10,).")
    ):
        graph.Evidence(fg_state=fg.fg_state, value=np.zeros(10))

    evidence = graph.Evidence(fg_state=fg.fg_state, value=np.zeros(15))
    assert jnp.all(evidence.value == jnp.zeros(15))

    vg2 = vgroup.VariableDict(variable_names=(0,), num_states=15)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got evidence for a variable or a VariableGroup not in the FactorGraph!"
        ),
    ):
        graph.update_evidence(
            jax.device_put(evidence.value),
            {vg2[0]: jax.device_put(np.zeros(15))},
            fg.fg_state,
        )


def test_bp():
    vg = vgroup.VariableDict(variable_names=(0,), num_states=15)
    fg = graph.FactorGraph(vg)
    factor_group = enumeration.EnumerationFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.add_factors(factor_group)

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
