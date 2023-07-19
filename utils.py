import random

import rasterio
import rasterio.features
import rasterio.warp
import os
import numpy as np
from matplotlib import pyplot
from rasterio.plot import show


def print_dataset_image(img):
    show(img.read([3, 2, 1]))


def print_mask(mask):
    pyplot.imshow(mask, cmap="pink")
    pyplot.title("mask")
    pyplot.show()
