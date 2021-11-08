# export
import sys

import matplotlib.pyplot as plt
import numpy as np
from learning import Model, index_to_rc, rc_to_index
from load_data import get_mnist_data_iters
from preproc import Preproc


# export
class Inference:
    def __init__(
        self,
        model,
        debug=False,
        num_iters=10,
        damping_factor=0.0,
        satisfaction_threshold=-10,
        score_type="default",
    ):
        self.model = model
        self.bp = BeliefPropogation(
            model,
            satisfaction_threshold=satisfaction_threshold,
            damping_factor=damping_factor,
            debug=debug,
            num_iters=num_iters,
            score_type=score_type,
        )

    def do(self, inf_img):
        self.bp.reset()
        for v in self.model.V:
            v.reset_beliefs()
        bu_msg = self.model.initialize_vertex_beliefs(inf_img)
        self.bp.bu_msg = bu_msg

        fs, vs = self.bp.loop()
        return fs, vs


import warnings

# export
from scipy.special import logsumexp

warnings.filterwarnings("error")


def get_max_indices(a):
    M = np.max(a)
    return np.argwhere(a == M)
    # return [[np.argmax(a)]]


class BeliefPropogation:
    def __init__(
        self,
        model,
        debug=False,
        num_iters=10,
        damping_factor=0.0,
        satisfaction_threshold=0.0,
        score_type="default",
    ):

        self.model = model
        self.factors = model.factors
        self.V = model.V

        self.damping_factor = damping_factor
        self.num_iters = num_iters
        self.debug = debug
        self.neg_inf = model.neg_inf
        self.score_type = score_type

        self.satisfaction_threshold = satisfaction_threshold
        self.reset()

    def reset(self):
        self._reset_incoming_msgs()
        self._iter = 0

    def _reset_incoming_msgs(self):
        self.incoming_msgs = [[] for i in range(len(self.V))]

    def _init_incoming_msgs(self):
        self.incoming_msgs = [[v.get_beliefs()] for v in self.V]

    def get_another_score(self):
        hps, vps = self.model.hps, self.model.vps
        max_msg = -1 + np.zeros_like(self.bu_msg)

        def _f(v, idx):
            f, r, c = v.get_frc()
            delta_r, delta_c = index_to_rc(idx, hps, vps)
            rd, cd = r + delta_r, c + delta_c
            # max_msg[f, rd-supress_radius:rd+supress_radius, cd-supress_radius:cd+supress_radius] = 1
            max_msg[f, rd, cd] = 1
            return

        for v in self.V:
            belief = v.get_beliefs()
            max_idxs = get_max_indices(belief)

            t = self.neg_inf
            for max_idx in max_idxs:
                _f(v, max_idx[0])

        return (max_msg == self.bu_msg).sum() - np.prod(self.bu_msg.shape)

    def get_score(self):
        factor_score = 0
        vertex_score = 0

        factors_satisfied = 0
        for factor in self.factors:
            i1, i2 = factor.i1, factor.i2
            belief1 = self.V[i1].get_beliefs()
            belief2 = self.V[i2].get_beliefs()

            max_idxs1 = get_max_indices(belief1)
            max_idxs2 = get_max_indices(belief2)

            t = self.neg_inf
            for max_idx1 in max_idxs1:
                for max_idx2 in max_idxs2:
                    idx1 = max_idx1[0]
                    idx2 = max_idx2[0]
                    t = max(t, factor.get_phi(idx1, idx2))

            if t >= self.satisfaction_threshold:
                factors_satisfied += 1
            factor_score += t

        vertices_satisfied = 0
        for v in self.V:
            belief = v.get_beliefs()
            max_idxs = get_max_indices(belief)

            t = self.neg_inf
            for max_idx in max_idxs:
                t = max(t, v.get_initial_beliefs()[max_idx[0]])

            if t >= self.satisfaction_threshold:
                vertices_satisfied += 1

            vertex_score += t

        return (
            factor_score,
            vertex_score,
            factors_satisfied / len(self.factors),
            vertices_satisfied / len(self.V),
        )

    def _run_one_iter(self):
        self._init_incoming_msgs()
        for factor in self.factors:
            i1, i2 = factor.i1, factor.i2
            belief1 = self.V[i1].get_beliefs()
            belief2 = self.V[i2].get_beliefs()

            incoming_msg = factor.compute_outgoing_msgs(belief1, belief2)
            self.incoming_msgs[i1].append(incoming_msg[i1])
            self.incoming_msgs[i2].append(incoming_msg[i2])

        self._merge_incoming_msgs()
        self.update_vertices()
        self._reset_incoming_msgs()

        self._iter += 1

        if self.debug:
            fs, vs, fg, vg = self.get_score()
            another_score = self.get_another_score()
            print(fs, vs, fg, vg, another_score)

    def _merge_incoming_msgs(self):
        for i, incoming_msg in enumerate(self.incoming_msgs):
            combined_belief = self._merge_incoming_msg(incoming_msg)
            self.incoming_msgs[i] = [combined_belief]

    def _merge_incoming_msg(self, incoming_msg):
        combined_belief = np.zeros(incoming_msg[0].shape)
        for b in incoming_msg:
            assert b.shape == incoming_msg[0].shape
            combined_belief = combined_belief + b

        ret = self._normalize(combined_belief)
        return ret

    def _normalize(self, arr):
        arr = arr - logsumexp(arr)
        arr[arr < self.neg_inf // 2] = self.neg_inf
        return arr

    def _update_belief(self, index):
        old_belief = self.V[index].get_beliefs()
        incoming_msg = self.incoming_msgs[index][0]

        assert old_belief.shape == incoming_msg.shape
        new_belief = old_belief + (1 - self.damping_factor) * (
            incoming_msg - old_belief
        )
        self.V[index].set_beliefs(new_belief)

    def update_vertices(self):
        for factor in self.factors:
            i1, i2 = factor.i1, factor.i2
            self._update_belief(i1)
            self._update_belief(i2)

    def loop(self):

        for i in range(self.num_iters):
            try:
                self._run_one_iter()
            except ValueError:
                print("divide by zero or other numerical precision error has occured!")
                return False, np.array(scores)

        fs, vs, fg, vg = self.get_score()
        another_score = self.get_another_score()

        if self.score_type == "default":
            return fs, vs
        return another_score, another_score

    def show_debug_images(self):
        if len(self.debug_images) == 0:
            return

        images_per_row = 5
        n_rows = int(ceil(self.num_iters / images_per_row))

        fig, axs = plt.subplots(n_rows, images_per_row, figsize=(15, 15))

        for i in range(len(self.debug_images)):
            if n_rows > 1:
                axs[i // images_per_row][i % images_per_row].imshow(
                    self.debug_images[i]
                )
            else:
                axs[i % images_per_row].imshow(self.debug_images[i], cmap="gray")
