from __future__ import annotations

import numpy as np
import torch
from datetime import datetime
from torch import optim
from matplotlib import pyplot as plt
import logging


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

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_name = (
            f'{self.output_label}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        )
        start_time = datetime.now()
        logging.info(f"Starting training model {model_name}")
        logging.info(f"Starting training at {start_time}")
        epoch_time = start_time
        for epoch in range(1, n_epochs + 1):
            # Sets model to train mode
            self.model.train()
            batch_losses = []
            for x_batch, value_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                x_batch = x_batch.to(self.device)
                loss = self.train_step(x_batch, value_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                self.model.eval()
                for x_batch, value_batch in val_loader:
                    # x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                    x_batch = x_batch.to(self.device)
                    predicted_value = self.model(x_batch)
                    loss = self.loss_fn(predicted_value, value_batch)
                    batch_val_losses.append(loss.item())
                validation_loss = np.mean(batch_val_losses)
                self.validation_losses.append(validation_loss)
            if True | (epoch <= 10) | (epoch % 50 == 0) | (epoch == n_epochs):
                logging.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f},{datetime.now()}, epoch time {datetime.now() - epoch_time}"
                )
            epoch_time = datetime.now()
        logging.info(
            f"Training finished at {datetime.now()}, total time {datetime.now() - start_time}"
        )
        torch.save(self.model.state_dict(), f"models/{model_name}.pt")
        return model_name

    def test(self, test_loader):
        with torch.no_grad():
            batch_losses = []
            self.model.eval()
            for x_test, value in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                value = value.to(self.device)
                predicted_value = self.model(x_test)
                loss = self.loss_fn(
                    predicted_value,
                    value,
                )
                batch_losses.append(loss.item())
            test_loss = np.mean(batch_losses)
            logging.info(f"Test loss: {test_loss:.4f}\t ")
        return test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.validation_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            values_predictions = []
            values = []
            for x_test, value in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                value = value.to(self.device)
                x_test = x_test.to(self.device)
                self.model.eval()
                predicted_value = self.model(x_test)
                values_predictions.append(
                    predicted_value.to(self.device).detach().numpy()
                )
                values.append(value.to(self.device).detach().numpy())

        return values_predictions, values

    def poll(self, x):
        x = x.unsqueeze(0).to(self.device)  # add batch dimension
        with torch.no_grad():
            self.model.eval()
            value = self.model(x)  # super slow during self play.
        return value.squeeze(0)  # remove batch dimension
