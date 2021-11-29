import numpy as np
from matplotlib import pyplot as plt


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def imshow(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def get_dataset_stats(dataset):
    """
    Calculates a dataset's stats
    :param dataset: corresponding to torch.Dataset
    :return: mean, std deviation
    """
    assert len(dataset) != 0

    total_sum = 0
    number_of_pixels = 0

    for image, _ in dataset:
        number_of_pixels += np.prod(image.shape)
        total_sum += np.sum(image.numpy())
    mean = total_sum/number_of_pixels

    variances_sum = 0
    for image, _ in dataset:
        variances_sum += np.sum((image.numpy() - mean) ** 2)
    stdev = np.sqrt(variances_sum/number_of_pixels)

    return mean, stdev
