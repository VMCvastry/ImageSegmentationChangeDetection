import torch
import argparse

from dataset import get_dataloaders
import logging

from dataset.dataset_utils import balance_dataset
from utils import getTrainer

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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--subset", type=float, default=1)
    parser.add_argument(
        "--dataset_location", type=str, default="./DynamicEarthNet_reduced"
    )
    parser.add_argument("--net_reduction", type=int, default=16)
    parser.add_argument("--net", type=str, default="unet")
    parser.add_argument("--val_accuracy", type=int, default=1)
    parser.add_argument("--load_model", type=str, default=None)

    args = parser.parse_args()
    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr
    subset_percentage = args.subset
    dataset_location = args.dataset_location
    net_reduction = args.net_reduction
    net = args.net
    val_accuracy = args.val_accuracy
    load_model = args.load_model
    logging.info(f"Parsed args: {args}")
    SEED = 42
    torch.manual_seed(SEED)
    train_loader, test_loader, val_loader, accuracy_loader = get_dataloaders(
        dataset_location,
        batch_size=batch_size,
        binary_change_detection=True,
        subset_percentage=subset_percentage,
        n_patches=4,
    )
    weight = balance_dataset(train_loader, True)
    trainer = getTrainer(
        net,
        net_reduction,
        lr,
        val_accuracy,
        weight,
        load_model=load_model,
    )
    # trainer.test(test_loader)

    trainer.train(
        train_loader, val_loader, batch_size=None, n_epochs=epochs, n_features=None
    )
    trainer.test(test_loader)
