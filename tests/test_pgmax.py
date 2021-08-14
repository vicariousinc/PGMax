from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

from pgmax.fg import graph, groups


def test_e2e_sanity_check():
    # Helper function to easily generate a list of valid configurations for a given suppression diameter
    def create_valid_suppression_config_arr(suppression_diameter):
        valid_suppressions_list = []
        base_list = [0] * suppression_diameter
        valid_suppressions_list.append(base_list)
        for idx in range(suppression_diameter):
            new_valid_list1 = base_list[:]
            new_valid_list2 = base_list[:]
            new_valid_list1[idx] = 1
            new_valid_list2[idx] = 2
            valid_suppressions_list.append(new_valid_list1)
            valid_suppressions_list.append(new_valid_list2)
        ret_arr = np.array(valid_suppressions_list)
        ret_arr.flags.writeable = False
        return ret_arr

    # Set random seed for rng
    rng = default_rng(23)
    true_final_msgs_output = jax.device_put(
        [
            0.0000000e00,
            -2.9802322e-08,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            -6.2903470e-01,
            0.0000000e00,
            -3.0177221e-01,
            -3.0177209e-01,
            -2.8220856e-01,
            0.0000000e00,
            -6.0928625e-01,
            -1.2259892e-01,
            -5.7227510e-01,
            0.0000000e00,
            -1.2259889e-01,
            0.0000000e00,
            -7.0657951e-01,
            0.0000000e00,
            -5.8416575e-01,
            -4.4967628e-01,
            -1.2259889e-01,
            -4.4967628e-01,
            0.0000000e00,
            0.0000000e00,
            -5.8398044e-01,
            -1.1587733e-01,
            -1.8589476e-01,
            0.0000000e00,
            -3.0177209e-01,
            0.0000000e00,
            -1.1920929e-07,
            -2.9802322e-08,
            0.0000000e00,
            0.0000000e00,
            -7.2534859e-01,
            -2.1119976e-01,
            -1.1224430e00,
            0.0000000e00,
            -2.9802322e-08,
            -2.9802322e-08,
            0.0000000e00,
            -1.1587762e-01,
            -4.8865837e-01,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            -2.9802322e-08,
            0.0000000e00,
            -1.7563977e00,
            -1.7563977e00,
            0.0000000e00,
            -2.0581698e00,
            -2.0581698e00,
            0.0000000e00,
            -2.1994662e00,
            -2.1994662e00,
            0.0000000e00,
            -1.6154857e00,
            -1.6154857e00,
            0.0000000e00,
            -2.0535989e00,
            -2.0535989e00,
            0.0000000e00,
            -1.7265215e00,
            -1.7265215e00,
            0.0000000e00,
            -1.7238994e00,
            -1.7238994e00,
            0.0000000e00,
            -2.0509768e00,
            -2.0509768e00,
            0.0000000e00,
            -1.8303051e00,
            -1.8303051e00,
            0.0000000e00,
            -2.7415483e00,
            -2.7415483e00,
            0.0000000e00,
            -2.0552459e00,
            -2.0552459e00,
            0.0000000e00,
            -2.1711233e00,
            -2.1711233e00,
        ]
    )

    # Create a synthetic depth image for testing purposes
    im_size = 3
    depth_img = 5.0 * np.ones((im_size, im_size))
    depth_img[
        np.tril_indices(im_size, 0)
    ] = 1.0  # This sets the lower triangle of the image to 1's
    depth_img = gaussian_filter(
        depth_img, sigma=0.5
    )  # Filter the depth image for realistic noise simulation?
    labels_img = np.zeros((im_size, im_size), dtype=np.int32)
    labels_img[np.tril_indices(im_size, 0)] = 1

    M: int = depth_img.shape[0]
    N: int = depth_img.shape[1]
    # Compute dI/dx (horizontal derivative)
    horizontal_depth_differences = depth_img[:-1] - depth_img[1:]
    # Compute dI/dy (vertical derivative)
    vertical_depth_differences = depth_img[:, :-1] - depth_img[:, 1:]
    # The below code block assigns values for the orientation of every horizontally-oriented cut
    # It creates a matrix to represent each of the horizontal cuts in the image graph (there will be (M-1, N) possible cuts)
    # Assigning a cut of 1 points to the left and a cut of 2 points to the right. A cut of 0 indicates an 'off; state
    horizontal_oriented_cuts = np.zeros((M - 1, N))
    horizontal_oriented_cuts[horizontal_depth_differences < 0] = 1
    horizontal_oriented_cuts[horizontal_depth_differences > 0] = 2
    # The below code block assigns values for the orientation of every vertically-oriented cut
    # It creates a matrix to represent each of the vertical cuts in the image graph (there will be (M-1, N) possible cuts)
    # Assigning a cut of 1 points up and a cut of 2 points down. A cut of 0 indicates an 'off' state
    vertical_oriented_cuts = np.zeros((M, N - 1))
    vertical_oriented_cuts[vertical_depth_differences < 0] = 1
    vertical_oriented_cuts[vertical_depth_differences > 0] = 2
    gt_has_cuts = np.zeros((2, M, N))
    gt_has_cuts[0, :-1] = horizontal_oriented_cuts
    gt_has_cuts[1, :, :-1] = vertical_oriented_cuts

    # Before constructing a Factor Graph, we specify all the valid configurations for the non-suppression factors
    valid_configs_non_supp = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [2, 0, 2, 0],
            [0, 0, 1, 1],
            [0, 0, 2, 2],
            [2, 0, 0, 1],
            [1, 0, 0, 2],
            [1, 0, 1, 1],
            [2, 0, 2, 1],
            [1, 0, 1, 2],
            [2, 0, 2, 2],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 2, 0],
            [1, 1, 0, 1],
            [2, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 2, 1],
            [1, 1, 1, 0],
            [2, 1, 2, 0],
            [0, 2, 0, 2],
            [2, 2, 0, 0],
            [0, 2, 1, 0],
            [2, 2, 0, 2],
            [1, 2, 0, 2],
            [0, 2, 2, 2],
            [0, 2, 1, 2],
            [2, 2, 2, 0],
            [1, 2, 1, 0],
        ]
    )
    valid_configs_non_supp.flags.writeable = False
    # Now, we specify the valid configurations for all the suppression factors
    SUPPRESSION_DIAMETER = 2
    valid_configs_supp = create_valid_suppression_config_arr(SUPPRESSION_DIAMETER)
    # We create a NDVariableArray such that the [0,i,j] entry corresponds to the vertical cut variable (i.e, the one
    # attached horizontally to the factor) that's at that location in the image, and the [1,i,j] entry corresponds to
    # the horizontal cut variable (i.e, the one attached vertically to the factor) that's at that location
    # We create a NDVariableArray such that the [0,i,j] entry corresponds to the vertical cut variable (i.e, the one
    # attached horizontally to the factor) that's at that location in the image, and the [1,i,j] entry corresponds to
    # the horizontal cut variable (i.e, the one attached vertically to the factor) that's at that location
    grid_vars_group = groups.NDVariableArray(3, (2, M - 1, N - 1))

    # Make a group of additional variables for the edges of the grid
    extra_row_keys: List[Tuple[Any, ...]] = [(0, row, N - 1) for row in range(M - 1)]
    extra_col_keys: List[Tuple[Any, ...]] = [(1, M - 1, col) for col in range(N - 1)]
    additional_keys = tuple(extra_row_keys + extra_col_keys)
    additional_keys_group = groups.GenericVariableGroup(3, additional_keys)

    # Combine these two VariableGroups into one CompositeVariableGroup
    composite_grid_group = groups.CompositeVariableGroup(
        {"grid_vars": grid_vars_group, "additional_vars": additional_keys_group}
    )

    gt_has_cuts = gt_has_cuts.astype(np.int32)

    # Now, we use this array along with the gt_has_cuts array computed earlier using the image in order to derive the evidence values
    grid_evidence_arr = np.zeros((2, M - 1, N - 1, 3), dtype=float)
    additional_vars_evidence_dict: Dict[Tuple[int, ...], np.ndarray] = {}
    for i in range(2):
        for row in range(M):
            for col in range(N):
                # The dictionary key is in composite_grid_group at loc [i,row,call]
                evidence_vals_arr = np.zeros(
                    3
                )  # Note that we know num states for each variable is 3, so we can do this
                evidence_vals_arr[
                    gt_has_cuts[i, row, col]
                ] = 2.0  # This assigns belief value 2.0 to the correct index in the evidence vector
                evidence_vals_arr = (
                    evidence_vals_arr - evidence_vals_arr[0]
                )  # This normalizes the evidence by subtracting away the 0th index value
                evidence_vals_arr[1:] += 0.1 * rng.logistic(
                    size=evidence_vals_arr[1:].shape
                )  # This adds logistic noise for every evidence entry
                try:
                    _ = composite_grid_group["grid_vars", i, row, col]
                    grid_evidence_arr[i, row, col] = evidence_vals_arr
                except ValueError:
                    try:
                        _ = composite_grid_group["additional_vars", i, row, col]
                        additional_vars_evidence_dict[(i, row, col)] = evidence_vals_arr
                    except ValueError:
                        pass

    # Create the factor graph
    fg = graph.FactorGraph(
        variable_groups=composite_grid_group,
    )

    # Set the evidence
    fg.set_evidence("grid_vars", grid_evidence_arr)
    fg.set_evidence("additional_vars", additional_vars_evidence_dict)

    # Create an EnumerationFactorGroup for four factors
    four_factor_keys: List[List[Tuple[Any, ...]]] = []
    for row in range(M - 1):
        for col in range(N - 1):
            if row != M - 2 and col != N - 2:
                four_factor_keys.append(
                    [
                        ("grid_vars", 0, row, col),
                        ("grid_vars", 1, row, col),
                        ("grid_vars", 0, row, col + 1),
                        ("grid_vars", 1, row + 1, col),
                    ]
                )
            elif row != M - 2:
                four_factor_keys.append(
                    [
                        ("grid_vars", 0, row, col),
                        ("grid_vars", 1, row, col),
                        ("additional_vars", 0, row, col + 1),
                        ("grid_vars", 1, row + 1, col),
                    ]
                )
            elif col != N - 2:
                four_factor_keys.append(
                    [
                        ("grid_vars", 0, row, col),
                        ("grid_vars", 1, row, col),
                        ("grid_vars", 0, row, col + 1),
                        ("additional_vars", 1, row + 1, col),
                    ]
                )
            else:
                four_factor_keys.append(
                    [
                        ("grid_vars", 0, row, col),
                        ("grid_vars", 1, row, col),
                        ("additional_vars", 0, row, col + 1),
                        ("additional_vars", 1, row + 1, col),
                    ]
                )

    # Create an EnumerationFactorGroup for vertical suppression factors
    vert_suppression_keys: List[List[Tuple[Any, ...]]] = []
    for col in range(N):
        for start_row in range(M - SUPPRESSION_DIAMETER):
            if col != N - 1:
                vert_suppression_keys.append(
                    [
                        ("grid_vars", 0, r, col)
                        for r in range(start_row, start_row + SUPPRESSION_DIAMETER)
                    ]
                )
            else:
                vert_suppression_keys.append(
                    [
                        ("additional_vars", 0, r, col)
                        for r in range(start_row, start_row + SUPPRESSION_DIAMETER)
                    ]
                )

    horz_suppression_keys: List[List[Tuple[Any, ...]]] = []
    for row in range(M):
        for start_col in range(N - SUPPRESSION_DIAMETER):
            if row != M - 1:
                horz_suppression_keys.append(
                    [
                        ("grid_vars", 1, row, c)
                        for c in range(start_col, start_col + SUPPRESSION_DIAMETER)
                    ]
                )
            else:
                horz_suppression_keys.append(
                    [
                        ("additional_vars", 1, row, c)
                        for c in range(start_col, start_col + SUPPRESSION_DIAMETER)
                    ]
                )

    # Make kwargs dicts
    # use this kwargs dict for 4 factors
    four_factors_kwargs = {
        "connected_var_keys": four_factor_keys,
        "factor_configs": valid_configs_non_supp,
    }
    vert_suppression_kwargs = {
        "connected_var_keys": vert_suppression_keys,
        "factor_configs": valid_configs_supp,
    }
    horz_suppression_kwargs = {
        "connected_var_keys": horz_suppression_keys,
        "factor_configs": valid_configs_supp,
    }

    fg.add_factors(four_factors_kwargs)
    fg.add_factors(vert_suppression_kwargs)
    fg.add_factors(horz_suppression_kwargs)

    # Run BP
    final_msgs = fg.run_bp(1000, 0.5)

    # Test that the output messages are close to the true messages
    assert jnp.allclose(final_msgs, true_final_msgs_output, atol=1e-06)
