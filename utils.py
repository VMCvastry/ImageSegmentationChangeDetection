import logging
import uuid

import numpy as np
import torch
from matplotlib import pyplot
from rasterio.plot import show

from trainer import Trainer
from unet_detection import get_model


def print_dataset_image(img):
    # show(img.read([3, 2, 1]))
    img = np.array(img.read([3, 2, 1, 4]))
    min_values = np.array([0, 0, 0, 0])
    max_values = np.array([0, 0, 0, 0])
    for i in range(4):
        min_values[i] = img[i, :, :].min()
        max_values[i] = img[i, :, :].max()
    normalized_img = np.zeros(img.shape, dtype=np.float32)
    for i in range(4):
        normalized_img[i, :, :] = (img[i, :, :] - min_values[i]) / (
            max_values[i] - min_values[i]
        )
    i = normalized_img[:3, :, :]
    i *= 255
    show(i.astype(np.uint8))


def print_mask(mask):
    pyplot.imshow(mask, cmap="pink")
    pyplot.title("mask")
    pyplot.show()


def show_confusion_image(pred: np.array, label: np.array):
    import numpy as np
    import matplotlib.pyplot as plt

    h, w = pred.shape

    # Create an empty image
    image = np.zeros((h, w, 3), dtype=np.uint8)
    tp = np.logical_and(pred == 1, label == 1)
    tn = np.logical_and(pred == 0, label == 0)
    fp = np.logical_and(pred == 1, label == 0)
    fn = np.logical_and(pred == 0, label == 1)

    logging.info(
        f"tp: {tp.sum() / (h * w) * 100:.2f}%, "
        f"tn: {tn.sum() / (h * w) * 100:.2f}%, "
        f"fp: {fp.sum() / (h * w) * 100:.2f}%, "
        f"fn: {fn.sum() / (h * w) * 100:.2f}%"
    )

    # Color TP, TN, FP, FN differently
    image[tp == 1] = [0, 255, 0]  # TP in green
    image[tn == 1] = [0, 0, 255]  # TN in blue
    image[fp == 1] = [255, 0, 0]  # FP in red
    image[fn == 1] = [255, 255, 0]  # FN in yellow

    # Display the image
    plt.imshow(image)
    plt.axis("off")  # Turn off axis labels and ticks
    plt.show()


def getTrainer(
    net,
    net_reduction,
    lr,
    val_accuracy,
    weight,
    load_model=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_model:
        if "$" in load_model:
            new_net = load_model.split("$")[0]
            logging.info(f"Changing net from {net} to {new_net}")
            net = new_net
    model = get_model(net, net_reduction)
    logging.info(f"Weight: {weight}")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(weight)]).to(device)
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
        output_label=f"{net}${str(uuid.uuid4())[0:8]}",
        load_model=load_model,
        loss_fn=criterion,
        optimizer=optimizer,
        val_accuracy=val_accuracy == 1,
    )
    return trainer
