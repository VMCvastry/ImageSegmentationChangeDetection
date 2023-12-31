import logging
import os
import time

import numpy as np
import rasterio
from constants import mean_dataset, std_dataset
from utils import print_dataset_image
import torch
from constants import SKIP_FACTOR


def unnormalize_img(img):
    img *= std_dataset.reshape((4, 1, 1))
    img += mean_dataset.reshape((4, 1, 1))
    img = img.astype(np.uint64)
    return img


def get_one_hot_from_mask(mask):
    one_hot = np.zeros((mask.shape[0], mask.shape[1], 43), dtype=np.int64)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            i = mask[x, y]
            one_hot[x, y, i] = 1
    return one_hot


def get_pairs(imgs: dict, labels: dict):
    # Creates all possible pairs of the 24 images for each zone.
    img_pairs = []
    label_pairs = []
    for key in imgs.keys():
        assert key in labels.keys()
        im = sorted(imgs[key])
        l = sorted(labels[key])
        skip = len(im) > 4
        for i, (img, label) in enumerate(zip(im, l)):
            if skip and i % SKIP_FACTOR != 0:
                continue
            for j, (img2, label2) in enumerate(zip(im, l)):
                if skip and (j + 1) % SKIP_FACTOR != 0:
                    continue
                img_pairs.append((img, img2))
                label_pairs.append((label, label2))

    return img_pairs, label_pairs


def load_img(path, show=False):
    f = rasterio.open(path)
    img = f.read()
    if show:
        print_dataset_image(f)
    # BGRN -> RGBN
    image = np.dstack((img[2], img[1], img[0], img[3]))
    return image
    # return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def get_patch(image, patch_number, n=16):
    patches_per_side = int(n**0.5)
    patch_size = image.shape[0] // patches_per_side

    row = (patch_number // patches_per_side) * patch_size  # Row start index
    col = (patch_number % patches_per_side) * patch_size  # Col start index

    patch = image[row : row + patch_size, col : col + patch_size]
    return patch


def get_img_files(path):
    zones = os.listdir(path)
    images = {}
    for zone in zones:
        images[zone] = []
        zone_path = os.path.join(path, zone)
        for img in os.listdir(zone_path):
            if img.endswith(".tif"):
                images[zone].append(os.path.join(zone_path, img))
    if "6813_3313_13" in images:
        del images["6813_3313_13"]
    # dummy = {
    #     "1311_3077_13": [
    #         os.path.join(path, "1311_3077_13/2018-01-01.tif"),
    #         os.path.join(path, "1311_3077_13/2018-02-01.tif"),
    #     ]
    # }
    return images


def get_labels_files(path):
    zones = os.listdir(path)
    labels = {}
    for zone in zones:
        zone_name = zone[:-4]
        labels[zone_name] = []
        zone_path = os.path.join(path, zone, "Labels/Raster")
        folder = os.listdir(zone_path)
        assert len(folder) == 1
        zone_path = os.path.join(zone_path, folder[0])
        for img in os.listdir(zone_path):
            if img.endswith(
                ".tif"
            ):  # There are a couple of .aux.xml files in the folder. eg 36N-30E-7N-L3H-SR-2018-06-01.tif.aux.xml
                labels[zone_name].append(os.path.join(zone_path, img))
    if "6813_3313_13" in labels:
        del labels["6813_3313_13"]

    dummy = {
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
    return labels


def get_computed_labels_files(pairs, path, binary_change_detection):
    # Read the precomputed change label for a given pair of labels
    path = os.path.join(path, f"computed_labels_b-{binary_change_detection}")
    labels = [f"{os.path.basename(i1)}_{os.path.basename(i2)}.npy" for i1, i2 in pairs]
    return [os.path.join(path, label) for label in labels]


def detect_data(root) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    # Get all files in dataset folder
    images_sources = get_img_files(os.path.join(root, "planet_reduced"))
    labels_sources = get_labels_files(os.path.join(root, "labels"))
    images_sources = {
        zone: images
        for zone, images in images_sources.items()
        if zone in labels_sources.keys()
    }  # 20 zones are not in the labels
    return images_sources, labels_sources


def balance_dataset_binary(train_loader):
    labels_positive = 0
    total_elements = 0

    for _, label in train_loader.dataset:
        labels_positive += torch.sum(label == 1).item()
        total_elements += torch.numel(label)

    labels_negative = total_elements - labels_positive

    logging.info(
        f"Positive labels: {labels_positive}/{labels_negative}, {labels_positive * 100 / (labels_negative + labels_positive)}"
    )
    return labels_negative / labels_positive


def balance_dataset_seg(train_loader):
    class_counts = torch.zeros(
        43, dtype=torch.float32
    )  # Initialize a counter for each class

    for (
        _,
        label,
    ) in (
        train_loader.dataset
    ):  # Assuming the label is a single channel with class IDs for each pixel
        for class_id in range(43):
            class_counts[class_id] += torch.sum(
                label == class_id
            ).item()  # Count pixels for each class

    total_elements = torch.sum(class_counts).item()
    class_weights = total_elements / class_counts

    max_weight = 1.0 * 10**4  # clamp extremely rare classes
    class_weights = torch.clamp(class_weights, max=max_weight)

    median_weight = torch.median(class_weights)
    normalized_class_weights = class_weights / median_weight  # normalize weights

    # # Handling the case when class_counts[class_id] is 0, to avoid nan values in class_weights
    # class_weights[class_counts == 0] = 1

    logging.info(
        f"Class counts: {class_counts.tolist()} \nWeights: {normalized_class_weights.tolist()}"
    )

    mean_weight = torch.mean(class_weights)
    test_weights = class_weights / mean_weight  # normalize weights
    logging.info(f"Weights with mean: {test_weights.tolist()}")
    return normalized_class_weights


def balance_dataset(train_loader, binary):
    start = time.time()
    if not binary:
        weights = balance_dataset_seg(train_loader)
    else:
        weights = balance_dataset_binary(train_loader)
    logging.info(f"Balance dataset took {time.time() - start:.2f}s")
    return weights


if __name__ == "__main__":
    # mgs = get_img_files("./DynamicEarthNet/planet_reduced")
    # labels = get_labels_files("./DynamicEarthNet/labels")
    pass
