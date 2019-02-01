import numpy as np
from albumentations import Resize, RandomGamma, Rotate, Compose, HueSaturationValue


def get_aug():
    return Compose([
        HueSaturationValue(p=0.7),
        RandomGamma(p=0.7),
        Rotate(limit=(-20,20),p=0.7),
        Resize(width=224, height=224)
    ]), Compose([
        Resize(width=224, height=224)
    ])


def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l, p in zip(labels, predictions)])
