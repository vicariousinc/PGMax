import re
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgmax import factor, fgraph, fgroup, infer, vgroup


def test_factor_graph():
    vg = vgroup.VarDict(variable_names=(0,), num_states=15)
    fg = fgraph.FactorGraph(vg)

    enum_factor = factor.EnumFactor(
        variables=[vg[0]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )
    fg.add_factors(enum_factor)

    factor_group = fgroup.EnumFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A Factor of type {factor.EnumFactor} involving variables {frozenset([(vg.__hash__(), 15)])} already exists."
        ),
    ):
        fg.add_factors(factor_group)


def test_bp_state():
    vg = vgroup.VarDict(variable_names=(0,), num_states=15)
    fg0 = fgraph.FactorGraph(vg)
    enum_factor = factor.EnumFactor(
        variables=[vg[0]],
        factor_configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
    )
    fg0.add_factors(enum_factor)

    fg1 = fgraph.FactorGraph(vg)
    fg1.add_factors(enum_factor)

    with pytest.raises(
        ValueError,
        match="log_potentials, ftov_msgs and evidence should be derived from the same fg_state",
    ):
        infer.BPState(
            log_potentials=fg0.bp_state.log_potentials,
            ftov_msgs=fg1.bp_state.ftov_msgs,
            evidence=fg1.bp_state.evidence,
        )


def test_log_potentials():
    vg = vgroup.VarDict(variable_names=(0,), num_states=15)
    fg = fgraph.FactorGraph(vg)
    factor_group = fgroup.EnumFactorGroup(
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
        factor_group2 = fgroup.EnumFactorGroup(
            variables_for_factors=[[vg[0]]],
            factor_configs=np.arange(10)[:, None],
        )
        fg.bp_state.log_potentials[factor_group2] = jnp.zeros((1, 15))

    with pytest.raises(
        ValueError,
        match=re.escape("Invalid FactorGroup queried to access log potentials."),
    ):
        fg.bp_state.log_potentials[vg[0]]

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials shape (10,). Got (15,)")
    ):
        infer.LogPotentials(fg_state=fg.fg_state, value=np.zeros(15))

    log_potentials = infer.LogPotentials(fg_state=fg.fg_state, value=np.zeros(10))
    assert jnp.all(log_potentials[factor_group] == jnp.zeros(10))


def test_ftov_msgs():
    vg = vgroup.VarDict(variable_names=(0,), num_states=15)
    fg = fgraph.FactorGraph(vg)
    factor_group = fgroup.EnumFactorGroup(
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
            f"Given belief shape (10,) does not match expected shape (15,) for variable ({vg.__hash__()}, 15)."
        ),
    ):
        fg.bp_state.ftov_msgs[vg[0]] = np.ones(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected messages shape (15,). Got (10,)")
    ):
        infer.FToVMessages(fg_state=fg.fg_state, value=np.zeros(10))

    ftov_msgs = infer.FToVMessages(fg_state=fg.fg_state, value=np.zeros(15))
    with pytest.raises(
        TypeError, match=re.escape("'FToVMessages' object is not subscriptable")
    ):
        ftov_msgs[(10,)]


def test_evidence():
    vg = vgroup.VarDict(variable_names=(0,), num_states=15)
    fg = fgraph.FactorGraph(vg)
    factor_group = fgroup.EnumFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.add_factors(factor_group)

    with pytest.raises(
        ValueError, match=re.escape("Expected evidence shape (15,). Got (10,).")
    ):
        infer.Evidence(fg_state=fg.fg_state, value=np.zeros(10))

    evidence = infer.Evidence(fg_state=fg.fg_state, value=np.zeros(15))
    assert jnp.all(evidence.value == jnp.zeros(15))

    vg2 = vgroup.VarDict(variable_names=(0,), num_states=15)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got evidence for a variable or a VarGroup not in the FactorGraph!"
        ),
    ):
        infer.bp_state.update_evidence(
            jax.device_put(evidence.value),
            {vg2[0]: jax.device_put(np.zeros(15))},
            fg.fg_state,
        )


def test_bp():
    vg = vgroup.VarDict(variable_names=(0,), num_states=15)
    fg = fgraph.FactorGraph(vg)
    factor_group = fgroup.EnumFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.add_factors(factor_group)

    bp = infer.BP(fg.bp_state, temperature=0)
    bp_arrays = bp.update()
    bp_arrays = bp.update(
        bp_arrays=bp_arrays,
        ftov_msgs_updates={vg[0]: np.zeros(15)},
    )
    bp_arrays = bp.run_bp(bp_arrays, num_iters=1)
    bp_arrays = replace(bp_arrays, log_potentials=jnp.zeros((10)))
    bp_state = bp.to_bp_state(bp_arrays)
    assert bp_state.fg_state == fg.fg_state
