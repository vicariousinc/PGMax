import matplotlib.pyplot as plt
import numpy as np
from preproc import Preproc
from scipy.special import logsumexp


def index_to_rc(index, hps, vps):
    return -hps + index // (2 * vps + 1), -vps + index % (2 * vps + 1)


def rc_to_index(r, c, hps, vps):
    return c + vps + (2 * hps + 1) * (r + hps)


def get_number_of_states(hps, vps):
    return (2 * hps + 1) * (2 * vps + 1)


def initialize_evidences(inf_img, frcs, hps, vps, neg_inf=-1000):
    M = get_number_of_states(hps, vps)

    preproc_layer = Preproc(cross_channel_pooling=True)
    bu_msg = preproc_layer.fwd_infer(inf_img)

    evidence_updates = {}

    for idx in range(frcs.shape[0]):
        frc = frcs[idx]

        unary_msg = neg_inf + np.zeros((frc.shape[0], M))

        for v in range(frc.shape[0]):

            f, r, c = frc[v, :]
            evidence = bu_msg[f, r - hps : r + hps + 1, c - hps : c + hps + 1]
            indices = np.transpose(np.nonzero(evidence > 0))

            for index in indices:
                r1, c1 = index
                delta_r, delta_c = r1 - hps, c1 - vps

                index = rc_to_index(delta_r, delta_c, hps, vps)
                unary_msg[v, index] = 0

        unary_msg = unary_msg - logsumexp(unary_msg, axis=1)[:, None]
        unary_msg[unary_msg < neg_inf // 2] = neg_inf

        evidence_updates[idx] = unary_msg

    return bu_msg, evidence_updates


def visualize_graph(frc, edge):
    img = np.zeros((200, 200))
    plt.figure(figsize=(10, 10))
    for e in edge:
        i1, i2, w = e
        f1, r1, c1 = frc[i1]
        f2, r2, c2 = frc[i2]

        img[r1, c1] = 255
        img[r2, c2] = 255
        plt.text((c1 + c2) // 2, (r1 + r2) // 2, str(w), color="blue")
        plt.plot([c1, c2], [r1, r2], color="blue", linewidth=0.5)

    plt.imshow(img, cmap="gray")
    return img
