from __future__ import annotations

"""Defines a logical factor"""

import functools
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import log_sigmoid, sigmoid

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
            The parent variable's state 1 is parents_edge_states[ii, 1] + 1.
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0,
            which takes into account all the LogicalFactors of the same subtype (OR/AND) of the FactorGraph.
            The child variable's state 1 is children_edge_states[ii, 1] + 1.
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1, for ANDFactors the edge_states_offset is -1.

    Raises:
        ValueError: If:
            (1) The are no num_logical_factors different factor indices
            (2) There is a factor index higher than num_logical_factors - 1
            (3) The edge_states_offset is not 1 or -1
    """

    parents_edge_states: Union[np.ndarray, jnp.ndarray]
    children_edge_states: Union[np.ndarray, jnp.ndarray]
    edge_states_offset: int

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

            if self.edge_states_offset != 1 and self.edge_states_offset != -1:
                raise ValueError(
                    f"The LogicalWiring's edge_states_offset must be 1 (for OR) and -1 (for AND), but is {self.edge_states_offset}"
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
            "edge_states_offset": self.edge_states_offset,
        }


@dataclass(frozen=True, eq=False)
class LogicalFactor(nodes.Factor):
    """A logical OR/AND factor of the form (p1,...,pn, c)
    where p1,...,pn are the parents variables and c is the child variable.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1, for ANDFactors the edge_states_offset is -1.

    Raises:
        ValueError: If:
            (1) There are less than 2 variables
            (2) The variables are not all binary
    """

    log_potentials: np.ndarray = field(init=False, default=np.empty((0,)))
    edge_states_offset: int = field(init=False)

    def __post_init__(self):
        if len(self.variables) < 2:
            raise ValueError(
                "At least one parent variable and one child variable is required"
            )

        if not np.all([variable.num_states == 2 for variable in self.variables]):
            raise ValueError("All variables should all be binary")

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
                edge_states_offset=1,
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
            edge_states_offset=wirings[0].edge_states_offset,
        )

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
        num_parents = len(self.variables) - 1
        relevant_state = (-self.edge_states_offset + 1) // 2
        parents_edge_states = np.vstack(
            [
                np.zeros(num_parents, dtype=int),
                np.arange(relevant_state, 2 * num_parents, 2, dtype=int),
            ],
        ).T
        child_edge_state = np.array([2 * num_parents + relevant_state], dtype=int)
        return LogicalWiring(
            edges_num_states=self.edges_num_states,
            var_states_for_edges=var_states_for_edges,
            parents_edge_states=parents_edge_states,
            children_edge_states=child_edge_state,
            edge_states_offset=self.edge_states_offset,
        )


@dataclass(frozen=True, eq=False)
class ORFactor(LogicalFactor):
    """An OR factor of the form (p1,...,pn, c)
    where p1,...,pn are the parents variables and c is the child variable.

    An OR factor is defined as:
    F(p1, p2, ..., pn, c) = 0 <=> c = OR(p1, p2, ..., pn)
    F(p1, p2, ..., pn, c) = -inf o.w.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1.
    """

    edge_states_offset: int = field(init=False, default=1)


@dataclass(frozen=True, eq=False)
class ANDFactor(LogicalFactor):
    """An AND factor of the form (p1,...,pn, c)
        where p1,...,pn are the parents variables and c is the child variable.

    An AND factor is defined as:
        F(p1, p2, ..., pn, c) = 0 <=> c = AND(p1, p2, ..., pn)
        F(p1, p2, ..., pn, c) = -inf o.w.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ANDFactors the edge_states_offset is -1.
    """

    edge_states_offset: int = field(init=False, default=-1)


import collections
from typing import FrozenSet, OrderedDict

from pgmax.fg.groups import FactorGroup
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class LogicalFactorGroup(FactorGroup):
    """Class to represent a group of LogicalFactors.

    All factors in the group are assumed to have the same edge_states_offset.
    Consequently, the factors are all ORFactors or ANDFactors.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1, for ANDFactors the edge_states_offset is -1.
    """

    log_potentials: np.ndarray = field(init=False, default=np.empty((0,)))
    edge_states_offset: int = field(init=False)

    def __post_init__(self):
        # TODO: move all asserts from EnumerationFactor to here
        super().__post_init__()
        pass

    @cached_property
    def factor_group_log_potentials_full(self) -> np.ndarray:
        return self.log_potentials

    def compile_wiring(self, vars_to_starts) -> LogicalWiring:
        """Compile LogicalWiring for the LogicalFactorGroup

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
             LogicalWiring for the LogicalFactorGroup
        """
        relevant_state = (-self.edge_states_offset + 1) // 2

        var_states_for_edges = []
        for variable_and_num_states in self.variables_and_num_states:
            variable, num_states = variable_and_num_states
            this_var_states_for_edges = np.arange(
                vars_to_starts[variable], vars_to_starts[variable] + num_states
            )
            var_states_for_edges.append(this_var_states_for_edges)

        edges_num_states_cumsum = 0
        parents_edge_states = []
        children_edge_states = []
        for factor_idx, variable_names_for_factor in enumerate(
            self.variable_names_for_factors
        ):
            num_parents = len(variable_names_for_factor) - 1

            # Note: edges_num_states_cumsum correspomds to the factor_to_msgs_start for the LogicalFactor
            this_parents_edge_states = np.vstack(
                [
                    np.full(num_parents, fill_value=factor_idx, dtype=int),
                    np.arange(
                        edges_num_states_cumsum + relevant_state,
                        edges_num_states_cumsum + 2 * num_parents,
                        2,
                        dtype=int,
                    ),
                ],
            ).T
            this_child_edge_state = (
                edges_num_states_cumsum + 2 * num_parents + relevant_state
            )

            parents_edge_states.append(this_parents_edge_states)
            children_edge_states.append(this_child_edge_state)
            edges_num_states_cumsum += 2 * (num_parents + 1)

        return LogicalWiring(
            edges_num_states=self.factor_edges_num_states,
            var_states_for_edges=np.concatenate(var_states_for_edges),
            parents_edge_states=np.concatenate(parents_edge_states),
            children_edge_states=np.array(children_edge_states),
            edge_states_offset=self.edge_states_offset,
        )


@dataclass(frozen=True, eq=False)
class ORFactorGroup(LogicalFactorGroup):
    """Class to represent a group of ORFactors.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1.
    """

    edge_states_offset: int = field(init=False, default=1)

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, LogicalFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionary mapping all possible set of connected variables to different factors.
        """
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(self.variable_names_for_factors[ii]),
                    ORFactor(
                        variables=tuple(
                            self.variable_group[self.variable_names_for_factors[ii]]
                        ),
                    ),
                )
                for ii in range(len(self.variable_names_for_factors))
            ]
        )
        return variables_to_factors


@dataclass(frozen=True, eq=False)
class ANDFactorGroup(LogicalFactorGroup):
    """Class to represent a group of ANDFactors.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ANDFactors the edge_states_offset is -1.
    """

    edge_states_offset: int = field(init=False, default=-1)

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, LogicalFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionary mapping all possible set of connected variables to different factors.
        """
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(self.variable_names_for_factors[ii]),
                    ANDFactor(
                        variables=tuple(
                            self.variable_group[self.variable_names_for_factors[ii]]
                        ),
                    ),
                )
                for ii in range(len(self.variable_names_for_factors))
            ]
        )
        return variables_to_factors


@functools.partial(jax.jit, static_argnames=("temperature"))
def pass_logical_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    parents_edge_states: jnp.ndarray,
    children_edge_states: jnp.ndarray,
    edge_states_offset: int,
    temperature: float,
    log_potentials: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:

    """Passes messages from LogicalFactors to Variables.

    Args:
        vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened variable to all the LogicalFactors messages.
        parents_edge_states: Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the global LogicalFactor index,
            parents_edge_states[ii, 1] contains the message index of the parent variable's relevant state.
            For ORFactors the relevant state is 0, for ANDFactors the relevant state is 1.
            Both indices only take into account the LogicalFactors of the FactorGraph
            The parent variable's other state is parents_edge_states[ii, 1] + edge_states_offset
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's relevant state.
            For ORFactors the relevant state is 0, for ANDFactors the relevant state is 1.
            The child variable's other state is children_edge_states[ii, 1] + edge_states_offset
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1, for ANDFactors the edge_states_offset is -1.
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened ORFactors to variable messages.
    """
    num_factors = children_edge_states.shape[0]
    factor_indices = parents_edge_states[..., 0]

    parents_tof_msgs = (
        vtof_msgs[parents_edge_states[..., 1] + edge_states_offset]
        - vtof_msgs[parents_edge_states[..., 1]]
    )
    children_tof_msgs = (
        vtof_msgs[children_edge_states + edge_states_offset]
        - vtof_msgs[children_edge_states]
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
    ftov_msgs = ftov_msgs.at[parents_edge_states[..., 1] + edge_states_offset].set(
        parents_msgs
    )
    ftov_msgs = ftov_msgs.at[children_edge_states + edge_states_offset].set(
        children_msgs
    )
    return ftov_msgs
