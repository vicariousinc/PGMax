from __future__ import annotations

"""Defines a logical factor"""

import functools
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import log_sigmoid, sigmoid

from pgmax import utils
from pgmax.bp import bp_utils
from pgmax.fg import nodes


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class LogicalWiring(nodes.Wiring):
    """Wiring for LogicalFactors.

    Args:
        parents_edge_states: Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the global ORFactor index,
            parents_edge_states[ii, 1] contains the message index of the parent variable's state 0.
            Both indices only take into account the LogicalFactors of the same subtype (OR/AND) of the FactorGraph.
            The parent variable's state 1 is parents_edge_states[ii, 2] + 1.
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0,
            which takes into account all the LogicalFactors of the same subtype (OR/AND) of the FactorGraph.
            The child variable's state 1 is children_edge_states[ii, 1] + 1.

    Raises:
        ValueError: If:
            (1) The are no num_logical_factors different factor indices
            (2) There is a factor index higher than num_logical_factors - 1
    """

    parents_edge_states: Union[np.ndarray, jnp.ndarray]
    children_edge_states: Union[np.ndarray, jnp.ndarray]

    def __post_init__(self):
        if self.children_edge_states.shape[0] > 0:
            logical_factor_indices = self.parents_edge_states[:, 0]
            num_logical_factors = self.children_edge_states.shape[0]

            if np.unique(logical_factor_indices).shape[0] != num_logical_factors:
                raise ValueError(
                    f"The LogicalWiring must have {num_logical_factors} different LogicalFactor indices"
                )

            if logical_factor_indices.max() >= num_logical_factors:
                raise ValueError(
                    f"The highest LogicalFactor index must be {num_logical_factors - 1}"
                )

    @property
    def inference_arguments(self) -> Mapping[str, np.ndarray]:
        """
        Returns:
            A dictionnary of elements used to run belief propagation.
        """
        return {
            "parents_edge_states": self.parents_edge_states,
            "children_edge_states": self.children_edge_states,
        }


@dataclass(frozen=True, eq=False)
class LogicalFactor(nodes.Factor):
    """A logical OR/AND factor of the form
    p1    p2    p3   p4
    ||    ||    ||    ||
     \\   ||    ||   //
       \\  \\  //  //
             F
            ||
             c
    where p1... are the parents and c is the child.

    Raises:
        ValueError: If:
            (1) There are less than 2 variables
            (2) The variables are not all binary
    """

    log_potentials: np.ndarray = field(init=False, default=np.empty((0,)))

    def __post_init__(self):
        if len(self.variables) < 2:
            raise ValueError(
                "At least one parent variable and one child variable is required"
            )

        if not np.all([variable.num_states == 2 for variable in self.variables]):
            raise ValueError("All variables should all be binary")

    @utils.cached_property
    def parents_edge_states(self) -> np.ndarray:
        """
        Returns:
            Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the local ORFactor index,
            parents_edge_states[ii, 1] contains the message index of the parent variable's state 0.
        """
        num_parents = len(self.variables) - 1

        parents_edge_states = np.vstack(
            [
                np.zeros(num_parents, dtype=int),
                np.arange(0, 2 * num_parents, 2, dtype=int),
            ],
        ).T
        return parents_edge_states

    @utils.cached_property
    def child_edge_state(self) -> np.ndarray:
        """
        Returns:
            Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0.
        """
        return np.array([2 * (len(self.variables) - 1)], dtype=int)

    def compile_wiring(
        self, vars_to_starts: Mapping[nodes.Variable, int]
    ) -> LogicalWiring:
        """Compile LogicalWiring for the LogicalFactor

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
             LogicalWiring for the LogicalFactor
        """
        var_states_for_edges = np.concatenate(
            [
                np.arange(variable.num_states) + vars_to_starts[variable]
                for variable in self.variables
            ]
        )
        return LogicalWiring(
            edges_num_states=self.edges_num_states,
            var_states_for_edges=var_states_for_edges,
            parents_edge_states=self.parents_edge_states,
            children_edge_states=self.child_edge_state,
        )

    @staticmethod
    def concatenate_wirings(wirings: Sequence[LogicalWiring]) -> LogicalWiring:
        """Concatenate a list of LogicalWirings

        Args:
            wirings: A list of LogicalWirings

        Returns:
            Concatenated LogicalWiring
        """
        if len(wirings) == 0:
            return LogicalWiring(
                edges_num_states=np.empty((0,), dtype=int),
                var_states_for_edges=np.empty((0,), dtype=int),
                parents_edge_states=np.empty((0, 2), dtype=int),
                children_edge_states=np.empty((0,), dtype=int),
            )

        # Note: this correspomds to all the factor_to_msgs_starts for the LogicalFactors
        num_edge_states_cumsum = np.insert(
            np.array([wiring.edges_num_states.sum() for wiring in wirings]).cumsum(),
            0,
            0,
        )[:-1]

        parents_edge_states = []
        children_edge_states = []
        for ww, or_wiring in enumerate(wirings):
            offsets = np.array([[ww, num_edge_states_cumsum[ww]]], dtype=int)
            parents_edge_states.append(or_wiring.parents_edge_states + offsets)
            children_edge_states.append(or_wiring.children_edge_states + offsets[:, 1])

        return LogicalWiring(
            edges_num_states=np.concatenate(
                [wiring.edges_num_states for wiring in wirings]
            ),
            var_states_for_edges=np.concatenate(
                [wiring.var_states_for_edges for wiring in wirings]
            ),
            parents_edge_states=np.concatenate(parents_edge_states, axis=0),
            children_edge_states=np.concatenate(children_edge_states, axis=0),
        )


@dataclass(frozen=True, eq=False)
class ORFactor(LogicalFactor):
    """An OR factor of the form
    p1    p2    p3   p4
    ||    ||    ||    ||
     \\   ||    ||   //
       \\  \\  //  //
             F
            ||
             c
    where p1... are the parents and c is the child.

    An OR factor is defined as:
    F(p1, p2, ..., pn, c) = 0 <=> c = OR(p1, p2, ..., pn)
    F(p1, p2, ..., pn, c) = -inf o.w.
    """

    pass


@functools.partial(jax.jit, static_argnames=("temperature"))
def pass_OR_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    parents_edge_states: jnp.ndarray,
    children_edge_states: jnp.ndarray,
    temperature: float,
    log_potentials: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:

    """Passes messages from ORFactors to Variables.

    Args:
        vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened variable to all the ORFactors messages.
        parents_edge_states: Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the global ORFactor index,
            parents_edge_states[ii, 1] contains the message index of the parent variable's state 0.
            Both indices only take into account the ORFactors of the FactorGraph
            The parent variable's state 1 is parents_edge_states[ii, 2] + 1
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0
            The child variable's state 1 is children_edge_states[ii, 1] + 1
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened ORFactors to variable messages.
    """
    num_factors = children_edge_states.shape[0]

    factor_indices = parents_edge_states[..., 0]

    parents_tof_msgs = (
        vtof_msgs[parents_edge_states[..., 1] + 1]
        - vtof_msgs[parents_edge_states[..., 1]]
    )
    children_tof_msgs = (
        vtof_msgs[children_edge_states + 1] - vtof_msgs[children_edge_states]
    )

    # Consider the max-product case separately.
    # See https://arxiv.org/pdf/2111.02458.pdf, Appendix C.3
    if temperature == 0.0:
        # Get the first and second argmaxes for the incoming parents messages of each factor
        _, first_parents_argmaxes = bp_utils.get_maxes_and_argmaxes(
            parents_tof_msgs, factor_indices, num_factors
        )
        _, second_parents_argmaxes = bp_utils.get_maxes_and_argmaxes(
            parents_tof_msgs.at[first_parents_argmaxes].set(bp_utils.NEG_INF),
            factor_indices,
            num_factors,
        )

        parents_tof_msgs_pos = jnp.maximum(0.0, parents_tof_msgs)
        sum_parents_tof_msgs_pos = (
            jnp.full(shape=(num_factors,), fill_value=0.0)
            .at[factor_indices]
            .add(parents_tof_msgs_pos)
        )

        # Outgoing messages to parents variables
        parents_msgs = jnp.minimum(
            children_tof_msgs[factor_indices]
            + sum_parents_tof_msgs_pos[factor_indices]
            - parents_tof_msgs_pos,
            jnp.maximum(0.0, -parents_tof_msgs[first_parents_argmaxes][factor_indices]),
        )
        parents_msgs = parents_msgs.at[first_parents_argmaxes].set(
            jnp.minimum(
                children_tof_msgs
                + sum_parents_tof_msgs_pos
                - parents_tof_msgs_pos[first_parents_argmaxes],
                jnp.maximum(0.0, -parents_tof_msgs[second_parents_argmaxes]),
            )
        )

        # Outgoing messages to children variables
        children_msgs = sum_parents_tof_msgs_pos + jnp.minimum(
            0.0, parents_tof_msgs[first_parents_argmaxes]
        )
    else:

        def g(x):
            "Useful function to implement belief propagation with a temperature > 0"
            return jnp.where(
                x == 0.0,
                0.0,
                x + temperature * jnp.log(1.0 - jnp.exp(-x / temperature)),
            )

        log_sig_parents_tof_msgs = -temperature * log_sigmoid(
            -parents_tof_msgs / temperature
        )
        sum_log_sig_parents_tof_msgs = (
            jnp.full(shape=(num_factors,), fill_value=0.0)
            .at[factor_indices]
            .add(log_sig_parents_tof_msgs)
        )
        g_sum_log_sig_parents_minus_id = g(
            sum_log_sig_parents_tof_msgs[factor_indices] - log_sig_parents_tof_msgs
        )

        # Outgoing messages to parents variables
        parents_msgs = -temperature * jnp.log(
            sigmoid(g_sum_log_sig_parents_minus_id / temperature)
            + sigmoid(-g_sum_log_sig_parents_minus_id / temperature)
            * jnp.exp(-children_tof_msgs[factor_indices] / temperature)
        )

        # Outgoing messages to children variables
        children_msgs = g(sum_log_sig_parents_tof_msgs)

    # Special case: factors with a single parent
    num_parents = jnp.bincount(factor_indices, length=num_factors)
    first_elements = jnp.concatenate(
        [jnp.zeros(1, dtype=int), jnp.cumsum(num_parents)]
    )[:-1]
    parents_msgs = parents_msgs.at[first_elements].set(
        jnp.where(num_parents == 1, children_tof_msgs, parents_msgs[first_elements]),
    )

    ftov_msgs = jnp.zeros_like(vtof_msgs)
    ftov_msgs = ftov_msgs.at[parents_edge_states[..., 1] + 1].set(parents_msgs)
    ftov_msgs = ftov_msgs.at[children_edge_states + 1].set(children_msgs)
    return ftov_msgs
