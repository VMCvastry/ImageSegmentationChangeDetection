import datetime

import torch

from dataset import get_dataloaders
from unet_detection import UNet


def train(model, train_loader, val_loader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = datetime.datetime.now()
    print(start)
    epoch_time = start
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} - Loss {total_loss / len(train_loader)}")
        print(datetime.datetime.now())
        print("Epoch time", datetime.datetime.now() - epoch_time)
        epoch_time = datetime.datetime.now()
        # torch.save(model.state_dict(), "./models/model.pth")
        # print("Saved model")
        with torch.no_grad():
            model.eval()
            total_loss = 0
            for i, (img, label) in enumerate(val_loader):
                output = model(img)
                loss = criterion(output, label)
                total_loss += loss.item()
            print(f"Validation loss: {total_loss / len(val_loader)}")
    print("Total time", datetime.datetime.now() - start)


def test(model, test_loader):
    model.eval()
    total_loss = 0
    for i, (img, label) in enumerate(test_loader):
        output = model(img)
        loss = criterion(output, label)
        total_loss += loss.item()

    print(f"Test loss: {total_loss / len(test_loader)}")


if __name__ == "__main__":
    # get args
    import argparse

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

    train_loader, test_loader, val_loader = get_dataloaders(
        dataset_location,
        batch_size=batch_size,
        binary_change_detection=True,
        subset_percentage=subset_percentage,
    )
    model = UNet(8, 1, reduction_factor=net_reduction)
    # model.load_state_dict(torch.load("./models/model.pth"))

    criterion = torch.nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, epochs=epochs, lr=lr)
    test(model, test_loader)
