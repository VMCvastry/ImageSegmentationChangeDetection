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
    net = "unet"
    val_accuracy = 1

    train_loader, test_loader, val_loader, accuracy_loader = get_dataloaders(
        dataset_location,
        batch_size=batch_size,
        binary_change_detection=True,
        subset_percentage=subset_percentage,
    )
    trainer = getTrainer(net, net_reduction, lr, val_accuracy, weight=[10.0])
    # trainer.test(test_loader)

    # trainer.train(
    #     train_loader, val_loader, batch_size=None, n_epochs=epochs, n_features=None
    # )
    # trainer.plot_losses()
    # trainer.test(test_loader)
    for x, y in test_loader:

        pred = trainer.poll(x)
        pred = torch.sigmoid(pred)
        pred = pred > 0.5
        print(pred.sum() / pred.numel(), y.sum() / y.numel())
        print((pred == y).sum() / pred.numel())

        show_confusion_image(pred.squeeze(0).numpy(), y.squeeze(0).numpy())
        input()
