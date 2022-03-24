import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, display=True, nr=None):
    "Useful function for visualizing several images"

    n_images, H, W = images.shape
    images = images - images.min()
    images /= images.max() + 1e-10

    if nr is None:
        nr = nc = np.ceil(np.sqrt(n_images)).astype(int)
    else:
        nc = n_images // nr
        assert n_images == nr * nc
    big_image = np.ones(((H + 1) * nr + 1, (W + 1) * nc + 1, 3))
    big_image[..., :3] = 0
    big_image[:: H + 1] = [0.502, 0, 0.502]
    im = 0
    for r in range(nr):
        for c in range(nc):
            if im < n_images:
                big_image[
                    (H + 1) * r + 1 : (H + 1) * r + 1 + H,
                    (W + 1) * c + 1 : (W + 1) * c + 1 + W,
                    :,
                ] = images[im, :, :, None]
            im += 1

    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(big_image, interpolation="none")
    return big_image
