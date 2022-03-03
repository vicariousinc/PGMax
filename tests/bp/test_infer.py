from itertools import product

import numpy as np

from pgmax.bp import infer

print("We assume that factor_id is increasing and take values in 0,..., n_factors")


def test_pass_fac_to_var_messages():
    """
    Compares the specialized pass_OR_fac_to_var_messages implementation which passes messages from OR Factors to Variables
    to the alternative pass_fac_to_var_messages which enumerates all the possible configurations.
    """

    for idx in range(10):
        np.random.seed(idx)

        # Define OR factor and incoming messages
        num_factors = np.random.randint(3, 6)
        num_parents = np.random.randint(1, 6, num_factors)
        vtof_msgs = np.random.normal(
            0, 1, size=(2 * (sum(num_parents) + len(num_parents)))
        )

        # Support for pass_fac_to_var_messages
        factor_configs_edge_states = None
        factor_config_start = 0
        edge_state_start = 0
        for this_num_parents in num_parents:
            configs = np.array(
                list(product([0, 1], repeat=this_num_parents + 1))
            ).astype(float)
            # Children state is last
            valid_ON_configs = configs[
                np.logical_and(configs[:, :-1].sum(axis=1) >= 1, configs[:, -1] == 1)
            ]
            valid_configs = np.concatenate(
                [np.zeros((1, this_num_parents + 1)), valid_ON_configs], axis=0
            )
            assert valid_configs.shape[0] == 2 ** this_num_parents

            positions = np.arange(
                edge_state_start, edge_state_start + 2 * (this_num_parents + 1), 2
            )
            edge_states = positions + valid_configs
            factor_config = np.arange(
                factor_config_start, factor_config_start + edge_states.shape[0]
            ).reshape(-1, 1) @ np.ones((1, this_num_parents + 1))
            this_factor_configs_edge_states = np.concatenate(
                [factor_config.reshape(-1, 1), edge_states.reshape(-1, 1)], axis=1
            )

            if factor_configs_edge_states is None:
                factor_configs_edge_states = this_factor_configs_edge_states
            else:
                factor_configs_edge_states = np.concatenate(
                    [factor_configs_edge_states, this_factor_configs_edge_states],
                    axis=0,
                )
            factor_config_start += 2 ** this_num_parents
            edge_state_start += 2 * (this_num_parents + 1)

        factor_configs_edge_states = factor_configs_edge_states.astype(np.int32)
        num_val_configs = factor_config_start
        log_potentials = np.zeros(num_val_configs)

        # Support for pass_OR_fac_to_var_messages
        parents_states = None
        edge_state_start = 0
        for factor_idx, this_num_parents in enumerate(num_parents):
            parents_positions = np.arange(
                edge_state_start, edge_state_start + 2 * this_num_parents, 2
            )
            this_parents_states = np.vstack(
                [np.full(parents_positions.shape[0], factor_idx), parents_positions]
            ).T
            this_children_states = np.array([edge_state_start + 2 * this_num_parents])

            if parents_states is None:
                parents_states = this_parents_states
                children_states = this_children_states
            else:
                parents_states = np.concatenate(
                    [parents_states, this_parents_states], axis=0
                )
                children_states = np.concatenate(
                    [children_states, this_children_states], axis=0
                )
            edge_state_start += 2 * (this_num_parents + 1)

        # With pass_fac_to_var_messages
        ftov_msgs1 = infer.pass_fac_to_var_messages(
            vtof_msgs, factor_configs_edge_states, log_potentials, num_val_configs, 0.0
        )
        ftoparents_msgs1 = (
            ftov_msgs1[parents_states[..., 1] + 1] - ftov_msgs1[parents_states[..., 1]]
        )
        ftochildren_msgs1 = (
            ftov_msgs1[children_states[..., 1] + 1]
            - ftov_msgs1[children_states[..., 1]]
        )

        # With pass_OR_fac_to_var_messages
        ftov_msgs2 = infer.pass_OR_fac_to_var_messages(
            vtof_msgs, parents_states, children_states
        )
        ftoparents_msgs2 = (
            ftov_msgs2[parents_states[..., 1] + 1] - ftov_msgs2[parents_states[..., 1]]
        )
        ftochildren_msgs2 = (
            ftov_msgs2[children_states[..., 1] + 1]
            - ftov_msgs2[children_states[..., 1]]
        )

        assert np.allclose(ftochildren_msgs1, ftochildren_msgs2, atol=1e-5)
        assert np.allclose(ftoparents_msgs1, ftoparents_msgs2, atol=1e-5)
