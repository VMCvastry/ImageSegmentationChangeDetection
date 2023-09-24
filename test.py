import random

import torch
import argparse

from dataset import get_dataloaders
import logging

from utils import show_confusion_image, getTrainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
)

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.setFormatter(formatter)
#
# file_handler = logging.FileHandler("logs.log")
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.addHandler(stdout_handler)


if __name__ == "__main__":
    batch_size = 1
    epochs = 2
    lr = 0.05
    subset_percentage = 1
    dataset_location = "./DynamicEarthNet_reduced"
    net_reduction = 16
    net = "sunet3"
    val_accuracy = 1
    SEED = random.randint(0, 100000)
    SEED = 42
    logging.info(f"Seed: {SEED}")
    torch.manual_seed(SEED)
    train_loader, test_loader, val_loader, accuracy_loader = get_dataloaders(
        dataset_location,
        batch_size=batch_size,
        binary_change_detection=True,
        subset_percentage=subset_percentage,
    )

    trainer = getTrainer(
        net,
        net_reduction,
        lr,
        val_accuracy,
        weight=10.0,
        # load_model="test_net_2c9a4e6b_2023-09-05_06-04-50", # sunet2
        # load_model="test_net_c18b5927_2023-09-05_23-15-49",  # unet 16
        # load_model="test_net_a07149d7_2023-09-06_02-48-22", # sunet3
        # load_model="eunet$ecc2d318_2023-09-06_06-44-37",  # eunet small
        # load_model="eunet$653f3c92_2023-09-06_07-28-59",  # eunet 10% p1
        load_model="eunet$506450b9_2023-09-06_18-31-01",  # eunet 10% p2
    )
    # trainer.test(test_loader)

    # trainer.train(
    #     train_loader, val_loader, batch_size=None, n_epochs=epochs, n_features=None
    # )
    # trainer.plot_losses()
    # trainer.test(test_loader)
    # torch.manual_seed(321214)
    for x, y in test_loader:
        if (y.sum() / y.numel()).item() < 0.1:
            continue

        pred = trainer.poll(x)
        pred = torch.sigmoid(pred)
        pred = pred > 0.5

        logging.info(
            f"label: {(y.sum() / y.numel()).item()*100:.2f}%, pred: {(pred.sum() / pred.numel()).item()*100:.2f}%"
        )
        logging.info(f"accuracy: {((pred == y).sum() / pred.numel()).item()*100:.2f}%")
        show_confusion_image(pred.squeeze(0).numpy(), y.squeeze(0).numpy())
        input()
