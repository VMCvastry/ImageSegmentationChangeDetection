import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from constants import mean_dataset, std_dataset
from dataset_utils import (
    get_pairs,
    load_img,
    detect_data,
    get_computed_labels_files,
    get_one_hot_from_mask,
)
from utils import print_mask


class DynamicEarthNet(Dataset):
    def __init__(self, root, binary_change_detection=True):
        self.binary_change_detection = binary_change_detection
        images_sources, labels_sources = detect_data(root)
        self.img_pairs, self.label_pairs = get_pairs(images_sources, labels_sources)
        print(f"Found {len(self.img_pairs)} pairs of images and labels")
        self.labels = get_computed_labels_files(
            self.label_pairs, root, binary_change_detection
        )
        self.mean = mean_dataset
        self.std = std_dataset
        self.normalize = transforms.Normalize(
            mean=np.concatenate((self.mean, self.mean)),
            std=np.concatenate((self.std, self.std)),
        )  # 8channels =(4+4)

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        img1 = load_img(self.img_pairs[index][0])
        img2 = load_img(self.img_pairs[index][1])
        img = np.concatenate((img1, img2), axis=2)  # 1024x1024x8    8=(4+4)
        img = np.transpose(img, (2, 0, 1))  # 8x1024x1024
        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img = self.normalize(img)
        label_file = self.labels[index]
        label = np.load(label_file)
        if not self.binary_change_detection:
            label = get_one_hot_from_mask(label)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return img, label


if __name__ == "__main__":
    d = DynamicEarthNet("./DynamicEarthNet", binary_change_detection=True)
    print(len(d))
    i1to2 = d[1]
    img, label = i1to2
    print_mask(label)
