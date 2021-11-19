# export
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# export
def get_mnist_data_iters(data_dir, train_size, test_size, full_test_set=False, seed=5):
    """
    Load MNIST data.

    Assumed data directory structure:
        training/
            0/
            1/
            2/
            ...
        testing/
            0/
            ...

    Parameters
    ----------
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    seed : int
        Random seed used by numpy.random for sampling training set.

    Returns
    -------
    train_data, train_data : [(numpy.ndarray, str)]
        Each item reps a data sample (2-tuple of image and label)
        Images are numpy.uint8 type [0,255]
    """
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num_per_class, get_filenames=False):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith("."):
                continue
            if num_per_class is None:
                samples = sorted(os.listdir(cat_path))
            else:
                samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class)

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                # Resize and pad the images to (200, 200)
                image_arr = cv2.resize(
                    src=plt.imread(filepath),
                    dsize=(112, 112),
                    interpolation=cv2.INTER_CUBIC,
                )
                img = np.pad(
                    image_arr,
                    pad_width=tuple([(p, p) for p in (44, 44)]),
                    mode="constant",
                    constant_values=0,
                )
                loaded_data.append((img, category))
        return loaded_data

    np.random.seed(seed)
    train_set = _load_data(
        os.path.join(data_dir, "training"), num_per_class=train_size // 10
    )
    test_set = _load_data(
        os.path.join(data_dir, "testing"),
        num_per_class=None if full_test_set else test_size // 10,
    )
    return train_set, test_set
