# export
from ctypes import CDLL, POINTER, byref, c_double, c_int, c_void_p

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.ctypeslib import ndpointer
from preproc import Preproc
from scipy.spatial import cKDTree, distance
from scipy.special import logsumexp


def index_to_rc(index, hps, vps):
    return -hps + (index - 1) // (2 * vps + 1), -vps + (index - 1) % (2 * vps + 1)


def rc_to_index(r, c, hps, vps):
    return 1 + c + vps + (2 * hps + 1) * (r + hps)


def get_number_of_states(hps, vps):
    return (2 * hps + 1) * (2 * vps + 1) + 1


class Vertex:
    def __init__(self, f, r, c, hps, vps, neg_inf=-1000):
        self.hps, self.vps = hps, vps
        self.f, self.r, self.c = f, r, c

        self.M = get_number_of_states(hps, vps)
        self.neg_inf = neg_inf
        self.reset_beliefs()

    def reset_beliefs(self):
        beliefs = [self.neg_inf] * self.M
        beliefs[0] = 0
        self.beliefs = np.array(beliefs)

    def normalize_beliefs(self):
        self.beliefs = self.beliefs - logsumexp(self.beliefs)
        self.beliefs[self.beliefs < self.neg_inf // 2] = self.neg_inf

    def update_belief_at_state(self, state, value):
        r, c = state
        if r > self.hps or r < -self.hps:
            raise ValueError(
                f"Please specify valid state[0] in the range [{-self.hps, self.hps}]"
            )
        if c > self.vps or c < -self.vps:
            raise ValueError(
                f"Please specify valid state[1] in the range [{-self.vps, self.vps}]"
            )

        index = rc_to_index(r, c, self.hps, self.vps)
        self.beliefs[index] = value

    def __repr__(self):
        return f"({self.f}, {self.r}, {self.c})"

    def __str__(self):
        return f"({self.f}, {self.r}, {self.c})"

    def get_initial_beliefs(self):
        return self.initial_beliefs

    def set_initial_beliefs(self):
        self.initial_beliefs = self.beliefs.copy() + 0.0

    def get_beliefs(self):
        return self.beliefs

    def set_beliefs(self, beliefs):
        self.beliefs = beliefs

    def get_frc(self):
        return self.f, self.r, self.c


class Factor:
    def __init__(
        self, i1, i2, perturb_radius, hps, vps, impl="pgmax", alpha=1.0, neg_inf=-1000
    ):
        if i1 == i2:
            raise ValueError("Cant have self-referecing factors")

        self.i1, self.i2 = i1, i2
        self.r = max(1, int(perturb_radius * alpha))
        self.hps, self.vps = hps, vps
        self.M = get_number_of_states(hps, vps)

        self.impl = impl
        self.neg_inf = neg_inf
        self._initialize_phi()

    def _initialize_phi(self):
        if self.impl == "pgmax":
            self._initialize_phi_py()
        elif self.impl == "cpp":
            self._initialize_phi_cpp()
        else:
            raise ValueError("Got invalid implementation type")

    def compute_outgoing_msgs(self, belief1, belief2):
        if self.impl == "pgmax":
            return self.compute_outgoing_msgs_py(belief1, belief2)
        elif self.impl == "cpp":
            return self.compute_outgoing_msgs_cpp(belief1, belief2)
        else:
            raise ValueError("Got invalid implementation type")

    def get_phi(self, i, j):
        if self.impl == "pgmax":
            return self.phi[i, j]
        elif self.impl == "cpp":
            return self.get_phi_cpp(i, j)
        else:
            raise ValueError("Got invalid implementation type")

    def get_phi_cpp(self, i, j):
        value = c_int()
        gp = self.lib.get_phi
        gp(c_int(i), c_int(j), self.rows, self.cols, self.size, byref(value))

        return value.value

    def _initialize_phi_py(self):
        index = 0
        rows = []
        cols = []
        vals = []
        for i in range(1, self.M):
            r1, c1 = index_to_rc(i, self.hps, self.vps)

            r2_min = max(r1 - self.r, -self.hps)
            r2_max = min(r1 + self.r, self.hps)
            c2_min = max(c1 - self.r, -self.vps)
            c2_max = min(c1 + self.r, self.vps)

            for r2 in range(r2_min, r2_max + 1):
                for c2 in range(c2_min, c2_max + 1):
                    j = rc_to_index(r2, c2, self.hps, self.vps)
                    rows.append(i)
                    cols.append(j)
                    vals.append(-self.neg_inf)
                    index += 1

        self.rows = rows
        self.cols = cols

        self.nnz = len(vals)

    def _initialize_phi_cpp(self):
        lib = CDLL("./init_matrix.so")
        rows = POINTER(c_int)()
        cols = POINTER(c_int)()
        size = c_int()

        lib._initialize_phi(
            self.M,
            int(self.r),
            self.hps,
            self.vps,
            byref(size),
            byref(rows),
            byref(cols),
        )
        self.rows, self.cols, self.size = rows, cols, size
        self.lib = lib
        self.nnz = size.value

    def compute_outgoing_msgs_py(self, belief1, belief2):
        msg2 = np.max(belief1 + self.phi, axis=1)
        msg1 = np.max(belief2 + self.phi, axis=1)

        return {self.i1: msg1, self.i2: msg2}

    def compute_outgoing_msgs_cpp(self, belief1, belief2):
        cnb = self.lib.compute_new_belief
        cnb.restype = ndpointer(dtype=c_double, shape=(self.M,))
        msg2 = cnb(
            self.rows,
            self.cols,
            self.size,
            c_void_p(belief1.ctypes.data),
            c_int(self.M),
        )
        msg1 = cnb(
            self.rows,
            self.cols,
            self.size,
            c_void_p(belief2.ctypes.data),
            c_int(self.M),
        )

        return {self.i1: msg1, self.i2: msg2}

    def get_valid_states(self):
        if self.impl == "cpp":
            rows = np.ctypeslib.as_array(self.rows, shape=(self.size.value,))
            cols = np.ctypeslib.as_array(self.cols, shape=(self.size.value,))
        elif self.impl == "pgmax":
            rows = self.rows
            cols = self.cols
        else:
            raise ValueError("Got invalid implementation type")

        return np.stack([rows, cols], axis=1)


def get_max_indices(a):
    M = np.max(a)
    return np.argwhere(a == M)
    # return [[np.argmax(a)]]


class Model:
    def __init__(
        self,
        img,
        hps,
        vps,
        num_orients=16,
        suppress_radius=3,
        perturb_factor=2.0,
        max_cxn_length=100,
        tolerance=4.0,
        alpha=1.0,
        factor_type="cpp",
        neg_inf=-1000,
    ):

        self.img = img
        self.shape = img.shape
        self.hps, self.vps = hps, vps

        self.num_orients = num_orients
        self.suppress_radius = suppress_radius
        self.perturb_factor = perturb_factor
        self.max_cxn_length = max_cxn_length

        self.factor_type = factor_type
        self.alpha = alpha
        self.tolerance = tolerance
        self.neg_inf = neg_inf

        self._init_vertices()
        self._init_factors()

    def _init_vertices(self):
        preproc_layer = Preproc(num_orients=self.num_orients)
        bu_msg = preproc_layer.fwd_infer(self.img)

        V = []
        img_edges = bu_msg.max(0) > 0
        # Compute the pools that are active (or sparsify)
        while True:
            r, c = np.unravel_index(img_edges.argmax(), img_edges.shape)
            if not img_edges[r, c]:
                break

            v = Vertex(
                bu_msg[:, r, c].argmax(),
                r,
                c,
                hps=self.hps,
                vps=self.vps,
                neg_inf=self.neg_inf,
            )
            V.append(v)

            img_edges[
                r - self.suppress_radius : r + self.suppress_radius + 1,
                c - self.suppress_radius : c + self.suppress_radius + 1,
            ] = False

        self.V = V

    def _init_factors(self):
        self.factors = []
        self.sum_log_z = 0

        graph = nx.Graph()
        graph.add_nodes_from([i for i, v in enumerate(self.V)])

        frcs = [(v.f, v.r, v.c) for v in self.V]
        frcs = np.array(frcs)

        f1_bus_tree = cKDTree(frcs[:, 1:])
        close_pairs = np.array(list(f1_bus_tree.query_pairs(r=self.max_cxn_length)))
        dists = [distance.euclidean(frcs[x, 1:], frcs[y, 1:]) for x, y in close_pairs]

        for close_pairs_idx in np.argsort(dists):
            source, target = close_pairs[close_pairs_idx]
            dist = dists[close_pairs_idx]

            try:
                perturb_dist = nx.shortest_path_length(
                    graph, source, target, "perturb_radius"
                )
            except nx.NetworkXNoPath:
                perturb_dist = np.inf

            target_perturb_dist = dist / float(self.perturb_factor)
            actual_perturb_dist = max(0, np.ceil(target_perturb_dist))
            if perturb_dist >= target_perturb_dist * self.tolerance:
                graph.add_edge(source, target, perturb_radius=int(actual_perturb_dist))

        total_rounding_error = 0
        for n1, n2 in nx.edge_dfs(graph):
            desired_radius = (
                distance.euclidean(frcs[n1, 1:], frcs[n2, 1:]) / self.perturb_factor
            )
            upper = int(np.ceil(desired_radius))
            lower = int(np.floor(desired_radius))
            round_up_error = total_rounding_error + upper - desired_radius
            round_down_error = total_rounding_error + lower - desired_radius

            if abs(round_up_error) < abs(round_down_error):
                graph.edges[n1, n2]["perturb_radius"] = upper
                total_rounding_error = round_up_error

                factor = Factor(
                    n1,
                    n2,
                    upper,
                    self.hps,
                    self.vps,
                    impl=self.factor_type,
                    alpha=self.alpha,
                    neg_inf=self.neg_inf,
                )

            else:
                graph.edges[n1, n2]["perturb_radius"] = lower
                total_rounding_error = round_down_error

                factor = Factor(
                    n1,
                    n2,
                    lower,
                    self.hps,
                    self.vps,
                    impl=self.factor_type,
                    alpha=self.alpha,
                    neg_inf=self.neg_inf,
                )

            self.sum_log_z += np.log(factor.nnz)
            self.factors.append(factor)

        del graph
        del frcs

    def initialize_vertex_beliefs(self, inf_img, debug=False):
        for v in self.V:
            v.reset_beliefs()

        preproc_layer = Preproc(cross_channel_pooling=True)
        bu_msg = preproc_layer.fwd_infer(inf_img)
        indices = np.transpose(np.nonzero(bu_msg > 0))

        if debug:
            img_edges = bu_msg.max(0) > 0
            plt.imshow(img_edges, cmap="gray")
            print(img_edges.shape)

        hps, vps = self.hps, self.vps
        for index in indices:
            f1, r1, c1 = index
            for v in self.V:
                f2, r2, c2 = v.get_frc()
                if abs(r1 - r2) > hps or abs(c1 - c2) > vps:
                    continue

                delta_r, delta_c = r1 - r2, c1 - c2

                v.update_belief_at_state((delta_r, delta_c), 0)
                v.beliefs[0] = self.neg_inf

        for v in self.V:
            v.normalize_beliefs()
            v.set_initial_beliefs()

        return bu_msg

    def visualize_graph(self):
        img = np.zeros(self.shape)

        for factor in self.factors:
            v1 = self.V[factor.i1]
            v2 = self.V[factor.i2]

            w = factor.r

            img[v1.r, v1.c] = 255
            img[v2.r, v2.c] = 255
            plt.text((v1.c + v2.c) // 2, (v1.r + v2.r) // 2, str(w), color="blue")
            plt.plot([v1.c, v2.c], [v1.r, v2.r], color="blue", linewidth=0.5)

        plt.imshow(img, cmap="gray")
