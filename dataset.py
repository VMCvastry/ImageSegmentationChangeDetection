import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_utils import (
    get_pairs,
    build_labels,
    load_img,
    get_img_files,
    get_labels_files,
    detect_data,
    get_computed_labels_files,
)
from utils import print_mask


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


class DynamicEarthNet(Dataset):
    def __init__(self, root, binary_change_detection=True):
        images_sources, labels_sources = detect_data(root)
        self.img_pairs, self.label_pairs = get_pairs(images_sources, labels_sources)
        print(f"Found {len(self.img_pairs)} pairs of images and labels")
        self.labels = get_computed_labels_files(
            self.label_pairs, root, binary_change_detection
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
        label_file = self.labels[index]
        label = np.load(label_file)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return img, label


if __name__ == "__main__":
    d = DynamicEarthNet("./DynamicEarthNet", binary_change_detection=True)
    print(len(d))
    i1to2 = d[1]
    img, label = i1to2
    print_mask(label)
