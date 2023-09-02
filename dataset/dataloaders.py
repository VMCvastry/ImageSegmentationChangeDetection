import torch
import torch.utils.data
from dataset import DynamicEarthNet

SEED = 42


def get_dataloaders(
    root, batch_size=4, binary_change_detection=True, subset_percentage=1
):
    # TODO make split deterministic

    data = DynamicEarthNet(root, binary_change_detection=binary_change_detection)
    data = torch.utils.data.Subset(data, range(int(subset_percentage * len(data))))
    train_size = int(0.6 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        data, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )
    val_size = int(0.5 * test_size)
    test_size = test_size - val_size
    test_dataset, val_dataset = torch.utils.data.random_split(
        test_dataset,
        [test_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(len(train_dataset), len(test_dataset), len(val_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=True,
    )
    accuracy_loader = torch.utils.data.DataLoader(
        data, batch_size=512, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=True,
    )
    return train_loader, test_loader, val_loader, accuracy_loader
