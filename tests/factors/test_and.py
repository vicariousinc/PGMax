from itertools import product

import jax
import numpy as np

from pgmax.fg import graph
from pgmax.groups import logical
from pgmax.groups import variables as vgroup


def test_run_bp_with_ANDFactors():
    """
    Simultaneously test
    (1) the support of ANDFactors in a FactorGraph and their specialized inference for different temperatures
    (2) the support of several factor types in a FactorGraph and during inference

    To do so, observe that an ANDFactor can be defined as an equivalent EnumerationFactor
    (which list all the valid OR configurations) and define two equivalent FactorGraphs
    FG1: first half of factors are defined as EnumerationFactors, second half are defined as ANDFactors
    FG2: first half of factors are defined as ANDFactors, second half are defined as EnumerationFactors

    Inference for the EnumerationFactors will be run with pass_enum_fac_to_var_messages while
    inference for the ANDFactors will be run with pass_logical_fac_to_var_messages.

    Note: for the first seed, add all the EnumerationFactors to FG1 and all the ANDFactors to FG2
    """
    for idx in range(10):
        print("it", idx)
        np.random.seed(idx)

        # Parameters
        num_factors = np.random.randint(3, 8)
        num_parents = np.random.randint(1, 6, num_factors)
        num_parents_cumsum = np.insert(np.cumsum(num_parents), 0, 0)

        # Setting the temperature
        if idx % 2 == 0:
            # Max-product
            temperature = 0.0
        else:
            temperature = np.random.uniform(low=0.5, high=1.0)

        # Graph 1
        parents_variables1 = vgroup.NDVariableArray(
            num_states=2, shape=(num_parents.sum(),)
        )
        children_variables1 = vgroup.NDVariableArray(num_states=2, shape=(num_factors,))
        fg1 = graph.FactorGraph(variables=[parents_variables1, children_variables1])

        # Graph 2
        parents_variables2 = vgroup.NDVariableArray(
            num_states=2, shape=(num_parents.sum(),)
        )
        children_variables2 = vgroup.NDVariableArray(num_states=2, shape=(num_factors,))
        fg2 = graph.FactorGraph(variables=[parents_variables2, children_variables2])

        # Variable names for factors
        variables_for_factors1 = []
        variables_for_factors2 = []
        for factor_idx in range(num_factors):
            variable_names1 = [
                parents_variables1[idx]
                for idx in range(
                    num_parents_cumsum[factor_idx],
                    num_parents_cumsum[factor_idx + 1],
                )
            ] + [children_variables1[factor_idx]]
            variables_for_factors1.append(variable_names1)

            variable_names2 = [
                parents_variables2[idx]
                for idx in range(
                    num_parents_cumsum[factor_idx],
                    num_parents_cumsum[factor_idx + 1],
                )
            ] + [children_variables2[factor_idx]]
            variables_for_factors2.append(variable_names2)

        # Option 1: Define EnumerationFactors equivalent to the ANDFactors
        for factor_idx in range(num_factors):
            this_num_parents = num_parents[factor_idx]

            configs = np.array(list(product([0, 1], repeat=this_num_parents + 1)))
            # Children state is last
            valid_AND_configs = configs[
                np.logical_and(
                    configs[:, :-1].sum(axis=1) < this_num_parents, configs[:, -1] == 0
                )
            ]
            valid_configs = np.concatenate(
                [np.ones((1, this_num_parents + 1), dtype=int), valid_AND_configs],
                axis=0,
            )
            assert valid_configs.shape[0] == 2**this_num_parents

            if factor_idx < num_factors // 2:
                # Add the first half of factors to FactorGraph1
                fg1.add_factor(
                    variable_names=variables_for_factors1[factor_idx],
                    factor_configs=valid_configs,
                    log_potentials=np.zeros(valid_configs.shape[0]),
                )
            else:
                if idx != 0:
                    # Add the second half of factors to FactorGraph2
                    fg2.add_factor(
                        variable_names=variables_for_factors2[factor_idx],
                        factor_configs=valid_configs,
                        log_potentials=np.zeros(valid_configs.shape[0]),
                    )
                else:
                    # Add all the EnumerationFactors to FactorGraph1 for the first iter
                    fg1.add_factor(
                        variable_names=variables_for_factors1[factor_idx],
                        factor_configs=valid_configs,
                        log_potentials=np.zeros(valid_configs.shape[0]),
                    )

        # Option 2: Define the ANDFactors
        num_parents_cumsum = np.insert(np.cumsum(num_parents), 0, 0)
        variables_for_ANDFactors_fg1 = []
        variables_for_ANDFactors_fg2 = []

        for factor_idx in range(num_factors):
            if factor_idx < num_factors // 2:
                # Add the first half of factors to FactorGraph2
                variables_for_ANDFactors_fg2.append(variables_for_factors2[factor_idx])
            else:
                if idx != 0:
                    # Add the second half of factors to FactorGraph1
                    variables_for_ANDFactors_fg1.append(
                        variables_for_factors1[factor_idx]
                    )
                else:
                    # Add all the ANDFactors to FactorGraph2 for the first iter
                    variables_for_ANDFactors_fg2.append(
                        variables_for_factors2[factor_idx]
                    )
        if idx != 0:
            fg1.add_factor_group(
                factory=logical.ANDFactorGroup,
                variables_for_factors=variables_for_ANDFactors_fg1,
            )
        fg2.add_factor_group(
            factory=logical.ANDFactorGroup,
            variables_for_factors=variables_for_ANDFactors_fg2,
        )

        # Run inference
        bp1 = graph.BP(fg1.bp_state, temperature=temperature)
        bp2 = graph.BP(fg2.bp_state, temperature=temperature)

        evidence_parents = jax.device_put(np.random.gumbel(size=(sum(num_parents), 2)))
        evidence_children = jax.device_put(np.random.gumbel(size=(num_factors, 2)))

        evidence_updates1 = {
            parents_variables1: evidence_parents,
            children_variables1: evidence_children,
        }
        evidence_updates2 = {
            parents_variables2: evidence_parents,
            children_variables2: evidence_children,
        }

        bp_arrays1 = bp1.init(evidence_updates=evidence_updates1)
        bp_arrays1 = bp1.run_bp(bp_arrays1, num_iters=5)
        bp_arrays2 = bp2.init(evidence_updates=evidence_updates2)
        bp_arrays2 = bp2.run_bp(bp_arrays2, num_iters=5)

        # Get beliefs
        beliefs1 = bp1.get_beliefs(bp_arrays1)
        beliefs2 = bp2.get_beliefs(bp_arrays2)

        assert np.allclose(
            beliefs1[children_variables1], beliefs2[children_variables2], atol=1e-4
        )
        assert np.allclose(
            beliefs1[parents_variables1], beliefs2[parents_variables2], atol=1e-4
        )
