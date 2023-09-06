from __future__ import annotations

import numpy as np
import torch
from datetime import datetime
from torch import optim
from matplotlib import pyplot as plt
import logging
from sklearn.metrics import f1_score

from constants import WEIGHT_POSITIVE


class Trainer:
    def __init__(
        self,
        *args,
        model,
        output_label: str,
        load_model: str = None,
        loss_fn=None,
        optimizer=None,
        val_accuracy=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        model = model.to(self.device)
        if load_model:
            model.load_state_dict(
                torch.load(f"models/{load_model}.pt", map_location=self.device)
            )
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.output_label = output_label
        self.train_losses = []
        self.validation_losses = []
        self.val_accuracy = val_accuracy

    def train_step(
        self,
        x,
        value,
    ):

        predicted_value = self.model(x)
        loss = self.loss_fn(
            predicted_value,
            value,
        )
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def batch_eval_cycle(self, loader, loss_fn=None, get_accuracy=False):
        if loss_fn is None:
            loss_fn = self.loss_fn
        losses = []
        self.model.eval()

        correct_proportional = 0
        positive = 0
        a = 0
        b = 0
        c = 0

        total_loss = 0
        correct = 0
        total = 0
        total_proportional = 0
        f1 = 0
        f1_c = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for x, label in loader:
            x, label = x.to(self.device), label.to(self.device, dtype=torch.float32)

            predicted_value = self.model(x)
            loss = loss_fn(predicted_value, label)
            total_loss += loss.item()

            if get_accuracy:
                flat_label = label.view(-1)
                a += predicted_value.sum().item()
                predicted_value = torch.sigmoid(predicted_value).view(-1)
                b += predicted_value.sum().item()
                predicted_value = predicted_value > 0.5
                c += predicted_value.sum().item()
                correct_map = predicted_value == flat_label
                correct += correct_map.sum().item()

                batch_positive = flat_label.sum().item()
                positive += batch_positive

                total_proportional += (  # POS * weight + NEG
                    batch_positive * WEIGHT_POSITIVE
                    + flat_label.size(0)
                    - batch_positive
                )
                correct_proportional += (  # Sum correct positive
                    correct_map * flat_label
                ).sum().item() * WEIGHT_POSITIVE
                correct_proportional += (
                    (correct_map & (flat_label == 0)).sum().item()
                )  # Sum correct negative

                TP += ((predicted_value == 1) & (flat_label == 1)).float().sum()
                TN += ((predicted_value == 0) & (flat_label == 0)).float().sum()
                FP += ((predicted_value == 1) & (flat_label == 0)).float().sum()
                FN += ((predicted_value == 0) & (flat_label == 1)).float().sum()

                f1_t = f1_score(
                    flat_label.cpu().numpy(),
                    predicted_value.cpu().numpy(),
                    zero_division=0,
                )

                if f1_t > 0:
                    f1 += f1_t
                    f1_c += 1

                total += flat_label.size(0)
            # break

        TPR = TP / (TP + FN) if TP + FN != 0 else 0
        TNR = TN / (TN + FP) if TN + FP != 0 else 0
        FPR = FP / (FP + TN) if FP + TN != 0 else 0
        FNR = FN / (FN + TP) if FN + TP != 0 else 0

        if get_accuracy:
            logging.info(f"{a / total}, {b / total}, {c / total}")
            logging.info(f"TPR: {TPR}, TNR: {TNR}, FPR: {FPR}, FNR: {FNR}")
            if f1_c > 0:
                logging.info(f"F1: {f1 / f1_c}, {f1_c}/{len(loader)}")
        return (
            total_loss / len(loader),
            correct / total if get_accuracy else -1,
            correct_proportional / total_proportional if get_accuracy else -1,
            positive / total if get_accuracy else -1,
        )

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_name = (
            f'{self.output_label}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        )
        start_time = datetime.now()
        logging.info(f"Starting training model {model_name}")
        logging.info(f"Starting training at {start_time}")
        epoch_time = start_time
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            batch_losses = []
            for x_batch, value_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                x_batch = x_batch.to(self.device)
                value_batch = value_batch.to(self.device)
                loss = self.train_step(x_batch, value_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            train_time = datetime.now() - epoch_time
            epoch_time = datetime.now()
            with torch.no_grad():
                validation_loss, accuracy, accuracy2, positive = self.batch_eval_cycle(
                    val_loader, get_accuracy=epoch % 5 == 0 or self.val_accuracy
                )
                self.validation_losses.append(validation_loss)
            if True | (epoch <= 10) | (epoch % 50 == 0) | (epoch == n_epochs):
                logging.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}, Accuracy: {accuracy*100:.2f}%, Accuracy (Proportional): {accuracy2*100:.2f}%, Positive {positive*100:.2f}%,{datetime.now()}, val time {datetime.now() - epoch_time}, train time {train_time}, total time {datetime.now() - epoch_time + train_time} "
                )
            epoch_time = datetime.now()
        logging.info(
            f"Training finished at {datetime.now()}, total time {datetime.now() - start_time}"
        )
        torch.save(self.model.state_dict(), f"models/{model_name}.pt")
        plot_train_losses(self.train_losses, self.validation_losses, self.output_label)
        return model_name

    def test(self, test_loader):
        with torch.no_grad():
            test_loss, accuracy, accuracy2, positive = self.batch_eval_cycle(
                test_loader, get_accuracy=True
            )
            logging.info(f"Test loss: {test_loss:.4f}\t ")
            logging.info(f"Accuracy of the network: {100 * accuracy:.2f}%")
            logging.info(
                f"Accuracy of the network (proportional): {100 * accuracy2:.2f}%"
            )
            logging.info(f"Positive: {100 * positive:.2f}%")

        return test_loss

    def poll(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            self.model.eval()
            value = self.model(x)
        return value.squeeze(0)  # remove batch dimension


def plot_train_losses(train_losses, validation_losses, output_label):
    data_mean = np.mean(train_losses)
    data_std = np.std(train_losses)
    num_std = 2
    min_value = data_mean - num_std * data_std
    max_value = data_mean + num_std * data_std

    plt.plot(np.clip(train_losses, min_value, max_value), label="Training loss")
    plt.plot(
        np.clip(validation_losses, min_value, max_value),
        label="Validation loss",
    )
    plt.legend()
    plt.title("Losses")
    # plt.show()
    plt.savefig(f"models/{output_label}_losses.png")
    plt.close()
