import datetime
import logging
import uuid

import torch
import argparse

from constants import WEIGHT_POSITIVE
from dataset import get_dataloaders
from unet_detection import UNet
from trainer import Trainer
import logging
import sys

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--subset", type=float, default=1)
    parser.add_argument(
        "--dataset_location", type=str, default="./DynamicEarthNet_reduced"
    )
    parser.add_argument("--net_reduction", type=int, default=32)
    args = parser.parse_args()
    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr
    subset_percentage = args.subset
    dataset_location = args.dataset_location
    net_reduction = args.net_reduction
    logging.info(f"Parsed args: {args}")

    train_loader, test_loader, val_loader, accuracy_loader = get_dataloaders(
        dataset_location,
        batch_size=batch_size,
        binary_change_detection=True,
        subset_percentage=subset_percentage,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(8, 1, reduction_factor=net_reduction)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(WEIGHT_POSITIVE * 2)]).to(device)
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     momentum=0.9,
    #     lr=lr,
    #     weight_decay=0.0001,
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(
        model=model,
        output_label=f"test_net_{str(uuid.uuid4())[0:8]}",
        load_model="",
        loss_fn=criterion,
        optimizer=optimizer,
    )
    trainer.test(test_loader)

    trainer.train(
        train_loader, val_loader, batch_size=None, n_epochs=epochs, n_features=None
    )
    trainer.plot_losses()
    trainer.test(test_loader)
