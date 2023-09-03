from __future__ import annotations

import numpy as np
import torch
from datetime import datetime
from torch import optim
from matplotlib import pyplot as plt
import logging

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
        correct = 0
        correct_proportional = 0
        total = 0
        total_proportional = 0
        positive = 0
        a = 0
        b = 0
        for x, label in loader:
            # x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
            x = x.to(self.device)
            label = label.to(self.device, dtype=torch.float32)
            predicted_value = self.model(x)
            loss = loss_fn(predicted_value, label)
            losses.append(loss.item())
            total += label.size(0) * label.size(1) * label.size(2)
            total_proportional += (  # POS * weight + NEG
                label.count_nonzero() * WEIGHT_POSITIVE
                + label.size(0) * label.size(1) * label.size(2)
                - label.count_nonzero()
            )
            if get_accuracy:
                a += predicted_value.sum().item()
                predicted_value = torch.sigmoid(predicted_value)
                predicted_value = predicted_value > 0.5
                b += predicted_value.sum().item()
                correct_map = predicted_value == label
                correct += correct_map.sum().item()
                positive += label.sum().item()
                correct_proportional += (  # Sum correct positive
                    correct_map * label
                ).sum().item() * WEIGHT_POSITIVE
                correct_proportional += (
                    (correct_map * (1 - label)).sum().item()
                )  # Sum correct negative
        # print(correct, total, correct_proportional, total_proportional)
        logging.info(f"{a / total}, {b / total}")
        loss = np.mean(losses)
        return (
            loss,
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

            with torch.no_grad():
                validation_loss, accuracy, accuracy2, positive = self.batch_eval_cycle(
                    val_loader, get_accuracy=True
                )
                self.validation_losses.append(validation_loss)
            if True | (epoch <= 10) | (epoch % 50 == 0) | (epoch == n_epochs):
                logging.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}, Accuracy: {accuracy*100:.2f}%, Accuracy (Proportional): {accuracy2*100:.2f}%, Positive {positive*100:.2f}%,{datetime.now()}, epoch time {datetime.now() - epoch_time}"
                )
            epoch_time = datetime.now()
        logging.info(
            f"Training finished at {datetime.now()}, total time {datetime.now() - start_time}"
        )
        torch.save(self.model.state_dict(), f"models/{model_name}.pt")
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

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.validation_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        # plt.show()
        plt.savefig(f"models/{self.output_label}_losses.png")
        plt.close()

    def poll(self, x):
        x = x.unsqueeze(0).to(self.device)  # add batch dimension
        with torch.no_grad():
            self.model.eval()
            value = self.model(x)
        return value.squeeze(0)  # remove batch dimension
