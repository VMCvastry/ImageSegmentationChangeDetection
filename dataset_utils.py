import numpy as np
import rasterio

from utils import print_dataset_image


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
