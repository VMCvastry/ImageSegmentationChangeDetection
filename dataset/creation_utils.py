import os

import numpy as np
import rasterio

from dataset.dataset_utils import (
    detect_data,
    load_img,
    get_pairs,
    get_img_files,
    get_labels_files,
)


def calculate_normalization_parameters(root):
    images_sources, labels_sources = detect_data(root)
    sum_channels = np.array([0, 0, 0, 0], dtype=np.float64)
    sum_squares_channels = np.array([0, 0, 0, 0], dtype=np.float64)
    total_pixels = 1024 * 1024 * len(images_sources) * 24
    c = 0
    for zone, images in list(images_sources.items()):
        c += 1
        print(
            f"Calculating normalization parameters for zone {c}/{len(images_sources)}"
        )
        for img_path in images:
            img = load_img(img_path)
            img = np.transpose(img, (2, 0, 1))  # C, H, W
            sum_channels += img.sum(axis=(1, 2))
    mean_channels = sum_channels / total_pixels
    c = 0
    for zone, images in list(images_sources.items()):
        c += 1
        print(
            f"Calculating normalization parameters for zone {c}/{len(images_sources)}"
        )
        for img_path in images:
            img = load_img(img_path)
            img = np.transpose(img, (2, 0, 1))  # C, H, W
            sum_squares_channels += ((img - mean_channels.reshape((4, 1, 1))) ** 2).sum(
                axis=(1, 2)
            )
    std_channels = np.sqrt(sum_squares_channels / total_pixels)
    print(mean_channels)
    print(std_channels)


def create_labels(root, binary_change_detection=True):
    images_sources, labels_sources = detect_data(root)
    img_pairs, label_pairs = get_pairs(images_sources, labels_sources)
    assert len(img_pairs) == len(label_pairs)
    print(f"Found {len(img_pairs)} pairs of images and labels")
    path = os.path.join(root, f"computed_labels_b-{binary_change_detection}")
    for i in range(0, len(label_pairs), 100):
        print(f"Building label {i}/{len(label_pairs)}", flush=True)
        labels = build_labels(label_pairs[i : i + 100], binary_change_detection)
        for name, label in labels.items():
            np.save(os.path.join(path, name), label)


def test_dataset_consistency():
    imgs = get_img_files("../DynamicEarthNet/planet_reduced")
    labels = get_labels_files("../DynamicEarthNet/labels")
    print(len(labels.keys()))
    print(len(imgs.keys()))
    for zone in imgs:
        if zone not in labels:
            # print("Zone {} not in labels".format(zone))
            continue
        print(zone)
        im = [x[-14:] for x in imgs[zone]]
        lb = [x[-14:] for x in labels[zone]]
        print(im)
        print(lb)
        eq = [x == y.replace("_", "-") for x, y in zip(im, lb)]
        assert all([x == y.replace("_", "-") for x, y in zip(im, lb)])
        print("")


def build_labels(label_pairs, binary_change_detection):
    labels = {}
    for i1, i2 in label_pairs:
        label = build_label((i1, i2), binary_change_detection)
        name = f"{os.path.basename(i1)}_{os.path.basename(i2)}.npy"
        if name in labels.keys():
            assert np.array_equal(labels[name], label)
        labels[name] = label
    return labels


def build_label(label_pair, binary_change_detection):
    i1, i2 = label_pair
    label1 = rasterio.open(i1).read()
    mask1 = build_label_mask(label1)
    label2 = rasterio.open(i2).read()
    mask2 = build_label_mask(label2)
    if binary_change_detection:
        label = build_change_mask(mask1, mask2)
    else:
        label = build_classification_mask(mask1, mask2)
    return label


def build_classification_mask(mask1, mask2):
    mask = np.zeros((mask1.shape[0], mask1.shape[1]), dtype=np.int64)
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
            mask[x, y] = i
    return mask


def get_one_hot_index(p1, p2):
    if p1 == p2:
        return 42
    i = 6 * p1 + (p2 if p2 < p1 else p2 - 1)
    return i


def build_label_mask(label):
    mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int64)
    for i in range(7):
        if i == 6:  # ignore the snow and ice class
            mask[label[i, :, :] == 255] = i  # NOTE Originally -1
        else:
            mask[label[i, :, :] == 255] = i
    return mask


def build_change_mask(mask1, mask2):
    mask = np.zeros((mask1.shape[0], mask1.shape[1]), dtype=np.int8)
    mask[mask1 == mask2] = 0
    mask[mask1 != mask2] = 1
    return mask


if __name__ == "__main__":
    # test_dataset_consistency()
    # create_labels("./DynamicEarthNet")
    calculate_normalization_parameters("./DynamicEarthNet")
