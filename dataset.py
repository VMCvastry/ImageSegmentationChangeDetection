import os

import numpy as np
import rasterio
import torch
from matplotlib import pyplot
from torch.utils.data import Dataset

from rasterio.plot import show

from utils import print_dataset_image, print_mask

def get_img_files(path):
    return {
        "1311_3077_13": [
            os.path.join(path, "1311_3077_13/2018-01-01.tif"),
            os.path.join(path, "1311_3077_13/2018-02-01.tif"),
        ]
    }


def get_labels_files(path):
    return {
        "1311_3077_13": [
            os.path.join(
                path,
                "1311_3077_13_10N/Labels/Raster/10N-122W-40N-L3H-SR/10N-122W-40N-L3H-SR-2018_01_01.tif",
            ),
            os.path.join(
                path,
                "1311_3077_13_10N/Labels/Raster/10N-122W-40N-L3H-SR/10N-122W-40N-L3H-SR-2018_02_01.tif",
            ),
        ]
    }


def get_pairs(imgs: dict, labels: dict):
    img_pairs = []
    label_pairs = []
    for key in imgs.keys():
        assert key in labels.keys()
        i = sorted(imgs[key])
        l = sorted(labels[key])
        for img, label in zip(i, l):
            for img2, label2 in zip(i, l):
                img_pairs.append((img, img2))
                label_pairs.append((label, label2))

    return img_pairs, label_pairs


def build_label_mask(label):
    mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int64)
    for i in range(7):
        if i == 6:  # ignore the snow and ice class
            mask[label[i, :, :] == 255] = i  # NOTE Originally -1
        else:
            mask[label[i, :, :] == 255] = i
    return mask


def build_change_mask(mask1, mask2):
    mask = np.zeros((mask1.shape[0], mask1.shape[1]), dtype=np.int64)
    mask[mask1 == mask2] = 0
    mask[mask1 != mask2] = 1
    return mask


def get_one_hot_index(p1, p2):
    if p1 == p2:
        return 42
    i = 6 * p1 + (p2 if p2 < p1 else p2 - 1)
    return i


def build_classification_mask(mask1, mask2):
    mask = np.zeros((mask1.shape[0], mask1.shape[1], 1), dtype=np.int64)
    p1, p2 = None, None
    # 0,0 0,1 0,2 0,3 0,4 0,5 0,6 1,0 1,1 1,2 1,3 1,4 1,5 1,6 2,0 2,1 2,2 2,3 2,4 2,5 2,6 3,0 3,1 3,2 3,3 3,4 3,5 3,6
    # X   0   1   2   3   4   5   6   X   7   8   9   10  11  12  13  X   14  15  16  17  18  19  20  X   21  22  23
    # Max=36+5=41
    # No change= 42
    for x in range(mask1.shape[0]):
        for y in range(mask1.shape[1]):
            p1 = mask1[x, y]
            p2 = mask2[x, y]
            i = get_one_hot_index(p1, p2)
            mask[x, y, 0] = i
    return mask


def build_labels(label_pairs, binary_change_detection):
    labels = []
    for i1, i2 in label_pairs:
        label1 = rasterio.open(i1).read()
        mask1 = build_label_mask(label1)
        label2 = rasterio.open(i2).read()
        mask2 = build_label_mask(label2)
        if binary_change_detection:
            label = build_change_mask(mask1, mask2)
        else:
            label = build_classification_mask(mask1, mask2)
        labels.append(label)
    return labels


def load_img(path):
    # read .tif
    f = rasterio.open(path)
    img = f.read()
    print_dataset_image(f)
    # BGRN -> RGBN
    image = np.dstack((img[2], img[1], img[0], img[3]))
    return image
    # return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


class DynamicEarthNet(Dataset):
    def __init__(self, root, binary_change_detection=True):
        self.images_sources = get_img_files(os.path.join(root, "planet_reduced"))
        self.labels_sources = get_labels_files(os.path.join(root, "labels"))
        assert len(self.images_sources) == len(self.labels_sources)
        self.img_pairs, self.label_pairs = get_pairs(
            self.images_sources, self.labels_sources
        )
        self.labels = build_labels(self.label_pairs, binary_change_detection)

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        img1 = load_img(self.img_pairs[index][0])
        img2 = load_img(self.img_pairs[index][1])
        img = np.concatenate((img1, img2), axis=2)  # 1024x1024x8    8=(4+4)
        img = np.transpose(img, (2, 0, 1))  # 8x1024x1024
        img = torch.from_numpy(np.array(img, dtype=np.float32))
        label = self.labels[index]
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return img, label


if __name__ == "__main__":
    d = DynamicEarthNet("./DynamicEarthNet", binary_change_detection=True)
    print(len(d))
    i1to2 = d[1]
    img, label = i1to2
    print_mask(label)
