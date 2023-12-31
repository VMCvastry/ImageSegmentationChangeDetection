import torch
import torch.utils.data
from dataset import DynamicEarthNet


def get_dataloaders(
    root, batch_size=4, binary_change_detection=True, subset_percentage=1, n_patches=16
):
    # TODO make split deterministic
    data = DynamicEarthNet(
        root, binary_change_detection=binary_change_detection, n_patches=n_patches
    )
    data = torch.utils.data.Subset(data, range(int(subset_percentage * len(data))))
    train_size = int(0.4 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        data,
        [train_size, test_size],
    )
    val_size = int(0.5 * test_size)
    test_size = test_size - val_size
    test_dataset, val_dataset = torch.utils.data.random_split(
        test_dataset,
        [test_size, val_size],
    )
    print(len(train_dataset), len(test_dataset), len(val_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    # balance_dataset(None, torch.cat([label for img, label in val_dataset]), True)
    # balance_dataset(None, torch.cat([label for img, label in test_dataset]), True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    accuracy_loader = torch.utils.data.DataLoader(
        data, batch_size=512, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    return train_loader, test_loader, val_loader, accuracy_loader
