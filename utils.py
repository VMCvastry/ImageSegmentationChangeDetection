import random

import rasterio
import rasterio.features
import rasterio.warp
import os
import numpy as np
from matplotlib import pyplot
from rasterio.plot import show


def print_dataset_image(img):
    # show(img.read([3, 2, 1]))
    img = np.array(img.read([3, 2, 1, 4]))
    min_values = np.array([0, 0, 0, 0])
    max_values = np.array([0, 0, 0, 0])
    for i in range(4):
        min_values[i] = img[i, :, :].min()
        max_values[i] = img[i, :, :].max()
    normalized_img = np.zeros(img.shape, dtype=np.float32)
    for i in range(4):
        normalized_img[i, :, :] = (img[i, :, :] - min_values[i]) / (
            max_values[i] - min_values[i]
        )
    i = normalized_img[:3, :, :]
    i *= 255
    show(i.astype(np.uint8))


def print_mask(mask):
    pyplot.imshow(mask, cmap="pink")
    pyplot.title("mask")
    pyplot.show()
