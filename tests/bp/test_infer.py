from itertools import product

import jax
import numpy as np

from pgmax.bp import infer
from pgmax.fg import graph, groups


def test_run_bp_with_OR_factors():
    """
    Tests the support of OR factor in a factor graph and the specialized inference by comparing two approaches:
    (1) Defining the equivalent enumeration factors of the OR factors (by listing all the valid configurations)
    and running inference with pass_fac_to_var_messages - which passes messages from enumeration factors to variables
    (2) Explicitly defining the ORFactor with and running the specialized pass_OR_fac_to_var_messages inference.
    """
    for idx in range(10):
        np.random.seed(idx)

        # Define OR factor and incoming messages
        num_factors = np.random.randint(3, 8)
        num_parents = np.random.randint(1, 6, num_factors)
        num_parents_cumsum = np.insert(np.cumsum(num_parents), 0, 0)

        # Setting the temperature
        if idx % 2 == 0:
            # Max-product
            temperature = 0.0
        else:
            temperature = np.random.uniform(low=0.5, high=1.0)

        # Graph 1: Defining OR factors as Enumeration Factor
        parents_variables1 = groups.NDVariableArray(
            num_states=2, shape=(num_parents.sum(),)
        )
        children_variable1 = groups.NDVariableArray(num_states=2, shape=(num_factors,))
        fg1 = graph.FactorGraph(
            variables=dict(parents=parents_variables1, children=children_variable1)
        )

        for factor_idx in range(num_factors):
            this_num_parents = num_parents[factor_idx]
            variable_names = [
                ("parents", idx)
                for idx in range(
                    num_parents_cumsum[factor_idx],
                    num_parents_cumsum[factor_idx + 1],
                )
            ] + [("children", factor_idx)]

            configs = np.array(list(product([0, 1], repeat=this_num_parents + 1)))
            # Children state is last
            valid_ON_configs = configs[
                np.logical_and(configs[:, :-1].sum(axis=1) >= 1, configs[:, -1] == 1)
            ]
            valid_configs = np.concatenate(
                [np.zeros((1, this_num_parents + 1), dtype=int), valid_ON_configs],
                axis=0,
            )
            assert valid_configs.shape[0] == 2 ** this_num_parents

            fg1.add_factor(
                factor_type="EnumerationFactor",
                variable_names=variable_names,
                factor_configs=valid_configs,
                log_potentials=np.zeros(valid_configs.shape[0]),
            )

        # Graph 2: Explicitly defining OR factors
        parents_variables2 = groups.NDVariableArray(
            num_states=2, shape=(num_parents.sum(),)
        )
        children_variable2 = groups.NDVariableArray(num_states=2, shape=(num_factors,))
        fg2 = graph.FactorGraph(
            variables=dict(parents=parents_variables2, children=children_variable2)
        )
        num_parents_cumsum = np.insert(np.cumsum(num_parents), 0, 0)
        variables_names_for_OR_factors = []

        for factor_idx in range(num_factors):
            variables_names_for_OR_factor = [
                ("parents", idx)
                for idx in range(
                    num_parents_cumsum[factor_idx],
                    num_parents_cumsum[factor_idx + 1],
                )
            ] + [("children", factor_idx)]
            variables_names_for_OR_factors.append(variables_names_for_OR_factor)

        fg2.add_factor_group(
            factory=groups.LogicalFactorGroup,
            variable_names_for_factors=variables_names_for_OR_factors,
            logical_type="OR",
        )

        # Test 1: Comparing both specialized inference functions
        vtof_msgs = np.random.normal(
            0, 1, size=(2 * (sum(num_parents) + len(num_parents)))
        )
        factor_configs_edge_states = fg1.fg_state.wiring.wiring_by_type[
            "EnumerationFactor"
        ].factor_configs_edge_states
        log_potentials = fg1.fg_state.log_potentials
        num_val_configs = int(factor_configs_edge_states[-1, 0]) + 1

        ftov_msgs1 = infer.pass_enum_fac_to_var_messages(
            vtof_msgs,
            factor_configs_edge_states,
            log_potentials,
            num_val_configs,
            temperature,
        )
        num_val_configs

        parents_edge_states = fg2.fg_state.wiring.wiring_by_type[
            "LogicalFactor"
        ].parents_edge_states
        children_edge_states = fg2.fg_state.wiring.wiring_by_type[
            "LogicalFactor"
        ].children_edge_states

        ftov_msgs2 = infer.pass_OR_fac_to_var_messages(
            vtof_msgs, parents_edge_states, children_edge_states, temperature
        )
        # Note: ftov_msgs1 and ftov_msgs2 are not normalized
        ftoparents_msgs1 = (
            ftov_msgs1[parents_edge_states[..., 1] + 1]
            - ftov_msgs1[parents_edge_states[..., 1]]
        )
        ftochildren_msgs1 = (
            ftov_msgs1[children_edge_states + 1] - ftov_msgs1[children_edge_states]
        )
        ftoparents_msgs2 = (
            ftov_msgs2[parents_edge_states[..., 1] + 1]
            - ftov_msgs2[parents_edge_states[..., 1]]
        )
        ftochildren_msgs2 = (
            ftov_msgs2[children_edge_states + 1] - ftov_msgs2[children_edge_states]
        )

        assert np.allclose(ftochildren_msgs1, ftochildren_msgs2, atol=1e-4)
        assert np.allclose(ftoparents_msgs1, ftoparents_msgs2, atol=1e-4)

        # Test 2: Running inference with graph.BP
        run_bp1, _, _ = graph.BP(fg1.bp_state, 1)
        run_bp2, _, _ = graph.BP(fg2.bp_state, 1)

        evidence_updates = {
            "parents": jax.device_put(np.random.gumbel(size=(sum(num_parents), 2))),
            "children": jax.device_put(np.random.gumbel(size=(num_factors, 2))),
        }

        bp_arrays1 = run_bp1(evidence_updates=evidence_updates)
        bp_arrays2 = run_bp2(evidence_updates=evidence_updates)
        assert np.allclose(bp_arrays1.ftov_msgs, bp_arrays2.ftov_msgs, atol=1e-4)
