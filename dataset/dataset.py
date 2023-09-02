import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from constants import mean_dataset, std_dataset, ONLINE_LABELS
from dataset.creation_utils import build_label
from .dataset_utils import (
    get_pairs,
    load_img,
    detect_data,
    get_computed_labels_files,
    get_one_hot_from_mask,
    balance_dataset,
)
from utils import print_mask


def get_label(files, root, binary_change_detection):
    if ONLINE_LABELS:
        return build_label(files, binary_change_detection)
    else:
        label_file = get_computed_labels_files(files, root, binary_change_detection)[0]
        return np.load(label_file)


class DynamicEarthNet(Dataset):
    def __init__(self, root, binary_change_detection=True):
        self.root = root
        self.binary_change_detection = binary_change_detection
        self.normalize = transforms.Normalize(
            mean=np.concatenate((mean_dataset, mean_dataset)),
            std=np.concatenate((std_dataset, std_dataset)),
        )  # 8channels =(4+4)
        images_sources, labels_sources = detect_data(root)
        self.img_pairs, self.label_pairs = get_pairs(images_sources, labels_sources)
        print(f"Found {len(self.img_pairs)} pairs of images and labels")
        # self.img_pairs, self.labels = balance_dataset(
        #     self.img_pairs, self.labels, self.binary_change_detection
        # )

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        img1 = load_img(self.img_pairs[index][0])
        img2 = load_img(self.img_pairs[index][1])
        img = np.concatenate((img1, img2), axis=2)  # 1024x1024x8    8=(4+4)
        img = np.transpose(img, (2, 0, 1))  # 8x1024x1024
        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img = self.normalize(img)
        label_files = self.label_pairs[index]
        label = get_label(label_files, self.root, self.binary_change_detection)
        # if not self.binary_change_detection:
        #     label = get_one_hot_from_mask(label)
        label = torch.from_numpy(np.array(label, dtype=np.float16))  # why float?
        return img, label


if __name__ == "__main__":
    d = DynamicEarthNet("./DynamicEarthNet_reduced", binary_change_detection=True)
    print(len(d))
    i1to2 = d[1]
    img, label = i1to2
    print_mask(label)
    pos = 0
    total = 0
    for img, label in d:
        pos += label.sum().item()
        total += label.size(0) * label.size(1)
    print(pos, total, pos / total)
