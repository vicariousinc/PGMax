from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

import pgmax
from pgmax.fg import graph as graph
from pgmax.interface import datatypes as interface_datatypes


def test_e2e_sanity_check():
    # Subclass FactorGroup into the 3 different groups that appear in this problem
    @dataclass
    class FourFactorGroup(interface_datatypes.FactorGroup):
        num_rows: int
        num_cols: int

        def connected_variables(
            self,
        ) -> List[List[Tuple[Any, ...]]]:
            ret_list: List[List[Tuple[Any, ...]]] = []
            for row in range(self.num_rows - 1):
                for col in range(self.num_cols - 1):
                    if row != self.num_rows - 2 and col != self.num_cols - 2:
                        ret_list.append(
                            [
                                ("grid_vars", 0, row, col),
                                ("grid_vars", 1, row, col),
                                ("grid_vars", 0, row, col + 1),
                                ("grid_vars", 1, row + 1, col),
                            ]
                        )
                    elif row != self.num_rows - 2:
                        ret_list.append(
                            [
                                ("grid_vars", 0, row, col),
                                ("grid_vars", 1, row, col),
                                ("additional_vars", 0, row, col + 1),
                                ("grid_vars", 1, row + 1, col),
                            ]
                        )
                    elif col != self.num_cols - 2:
                        ret_list.append(
                            [
                                ("grid_vars", 0, row, col),
                                ("grid_vars", 1, row, col),
                                ("grid_vars", 0, row, col + 1),
                                ("additional_vars", 1, row + 1, col),
                            ]
                        )
                    else:
                        ret_list.append(
                            [
                                ("grid_vars", 0, row, col),
                                ("grid_vars", 1, row, col),
                                ("additional_vars", 0, row, col + 1),
                                ("additional_vars", 1, row + 1, col),
                            ]
                        )

            return ret_list

    @dataclass
    class VertSuppressionFactorGroup(interface_datatypes.FactorGroup):
        num_rows: int
        num_cols: int
        suppression_diameter: int

        def connected_variables(
            self,
        ) -> List[List[Tuple[Any, ...]]]:
            ret_list: List[List[Tuple[Any, ...]]] = []
            for col in range(self.num_cols):
                for start_row in range(self.num_rows - self.suppression_diameter):
                    if col != self.num_cols - 1:
                        ret_list.append(
                            [
                                ("grid_vars", 0, r, col)
                                for r in range(
                                    start_row, start_row + self.suppression_diameter
                                )
                            ]
                        )
                    else:
                        ret_list.append(
                            [
                                ("additional_vars", 0, r, col)
                                for r in range(
                                    start_row, start_row + self.suppression_diameter
                                )
                            ]
                        )
            return ret_list

    @dataclass
    class HorzSuppressionFactorGroup(interface_datatypes.FactorGroup):
        num_rows: int
        num_cols: int
        suppression_diameter: int

        def connected_variables(
            self,
        ) -> List[List[Tuple[Any, ...]]]:
            ret_list: List[List[Tuple[Any, ...]]] = []
            for row in range(self.num_rows):
                for start_col in range(self.num_cols - self.suppression_diameter):
                    if row != self.num_rows - 1:
                        ret_list.append(
                            [
                                ("grid_vars", 1, row, c)
                                for c in range(
                                    start_col, start_col + self.suppression_diameter
                                )
                            ]
                        )
                    else:
                        ret_list.append(
                            [
                                ("additional_vars", 1, row, c)
                                for c in range(
                                    start_col, start_col + self.suppression_diameter
                                )
                            ]
                        )
            return ret_list

    # Override and define a concrete FactorGraph Class with the get_evidence function implemented
    class ConcreteFactorGraph(graph.FactorGraph):
        def get_evidence(
            self, data: Dict[pgmax.fg.nodes.Variable, np.array], context: Any = None
        ) -> jnp.ndarray:
            """Function to generate evidence array. Need to be overwritten for concrete factor graphs

            Args:
                data: Data for generating evidence
                context: Optional context for generating evidence

            Returns:
                None, but must set the self._evidence attribute to a jnp.array of shape (num_var_states,)
            """
            evidence = np.zeros(self.num_var_states)
            for var in self.variables:
                start_index = self._vars_to_starts[var]
                evidence[start_index : start_index + var.num_states] = data[var]
            return jax.device_put(evidence)

        def output_inference(
            self, final_var_states: jnp.ndarray, context: Any = None
        ) -> Any:
            """Function to take the result of message passing and output the inference result for
                each variable

            Args:
                final_var_states: an array of shape (num_var_states,) that is the result of belief
                    propagation
                context: Optional context for using this array

            Returns:
                An evidence array of shape (num_var_states,)
            """
            # NOTE: An argument can be passed here to do different inferences for sum-product and
            # max-product respectively
            var_to_map_dict = {}
            final_var_states_np = np.array(final_var_states)
            for var in self.variables:
                start_index = self._vars_to_starts[var]
                var_to_map_dict[var] = np.argmax(
                    final_var_states_np[start_index : start_index + var.num_states]
                )

            return var_to_map_dict

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
            -6.2903488e-01,
            0.0000000e00,
            -3.0177206e-01,
            -3.0177212e-01,
            -1.8688640e-01,
            0.0000000e00,
            -5.1396430e-01,
            -1.2259889e-01,
            -5.7227504e-01,
            0.0000000e00,
            -1.2259889e-01,
            0.0000000e00,
            -6.1125731e-01,
            0.0000000e00,
            -4.8884386e-01,
            -4.4967628e-01,
            -1.2259889e-01,
            -4.4967628e-01,
            0.0000000e00,
            0.0000000e00,
            -4.8865831e-01,
            -1.1587739e-01,
            -1.8589482e-01,
            0.0000000e00,
            -3.0177230e-01,
            0.0000000e00,
            -5.9604645e-08,
            -2.9802322e-08,
            0.0000000e00,
            0.0000000e00,
            -6.3002640e-01,
            -1.1587760e-01,
            -3.2707733e-01,
            0.0000000e00,
            0.0000000e00,
            -2.9802322e-08,
            -1.1920929e-07,
            -1.1587739e-01,
            -4.8865813e-01,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            -1.7563977e00,
            -1.7563977e00,
            0.0000000e00,
            -2.0581698e00,
            -2.0581698e00,
            0.0000000e00,
            -2.1041441e00,
            -2.1041441e00,
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
            -2.6462264e00,
            -2.6462264e00,
            0.0000000e00,
            -2.0552459e00,
            -2.0552459e00,
            0.0000000e00,
            -2.1711230e00,
            -2.9675078e-01,
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
    grid_vars_group = interface_datatypes.NDVariableArray(3, (2, M - 1, N - 1))
    # Make a group of additional variables for the edges of the grid
    extra_row_keys: List[Tuple[Any, ...]] = [(0, row, N - 1) for row in range(M - 1)]
    extra_col_keys: List[Tuple[Any, ...]] = [(1, M - 1, col) for col in range(N - 1)]
    additional_keys = tuple(extra_row_keys + extra_col_keys)
    additional_keys_group = interface_datatypes.GenericVariableGroup(3, additional_keys)
    # Combine these two VariableGroups into one CompositeVariableGroup
    composite_grid_group = interface_datatypes.CompositeVariableGroup(
        (("grid_vars", grid_vars_group), ("additional_vars", additional_keys_group))
    )

    # Now, we instantiate the four factors
    four_factors_group = FourFactorGroup(
        valid_configs_non_supp,
        composite_grid_group,
        M,
        N,
    )
    # Next, we instantiate all the vertical suppression variables
    vert_suppression_group = VertSuppressionFactorGroup(
        valid_configs_supp,
        composite_grid_group,
        M,
        N,
        SUPPRESSION_DIAMETER,
    )
    # Next, we instantiate all the horizontal suppression variables
    horz_suppression_group = HorzSuppressionFactorGroup(
        valid_configs_supp,
        composite_grid_group,
        M,
        N,
        SUPPRESSION_DIAMETER,
    )
    # Finally, we construct the tuple of all the factors and variables involved in the problem.
    facs_tuple = tuple(
        list(four_factors_group.factors)
        + list(vert_suppression_group.factors)
        + list(horz_suppression_group.factors)
    )
    vars_tuple = composite_grid_group.get_all_vars()

    # Test that these have the correct number of elements
    assert len(facs_tuple) == 10
    assert len(vars_tuple) == 12

    gt_has_cuts = gt_has_cuts.astype(np.int32)
    # Now, we use this array along with the gt_has_cuts array computed earlier using the image in order to derive the evidence values
    var_evidence_dict = {}
    for i in range(2):
        for row in range(M):
            for col in range(N):
                # The dictionary key is in composite_grid_group at loc [i,row,call]
                evidence_arr = np.zeros(
                    3
                )  # Note that we know num states for each variable is 3, so we can do this
                evidence_arr[
                    gt_has_cuts[i, row, col]
                ] = 2.0  # This assigns belief value 2.0 to the correct index in the evidence vector
                evidence_arr = (
                    evidence_arr - evidence_arr[0]
                )  # This normalizes the evidence by subtracting away the 0th index value
                evidence_arr[1:] += 0.1 * rng.logistic(
                    size=evidence_arr[1:].shape
                )  # This adds logistic noise for every evidence entry
                try:
                    var_evidence_dict[
                        composite_grid_group["grid_vars", i, row, col]
                    ] = evidence_arr
                except ValueError:
                    try:
                        var_evidence_dict[
                            composite_grid_group["additional_vars", i, row, col]
                        ] = evidence_arr
                    except ValueError:
                        pass

    # Create the factor graph
    fg = ConcreteFactorGraph(vars_tuple, facs_tuple)
    # Run BP
    final_msgs = fg.run_bp(1000, 0.5, evidence_data=var_evidence_dict)

    # Test that the output messages are close to the true messages
    assert jnp.allclose(final_msgs, true_final_msgs_output)
