from typing import Callable

import torch
from torch.utils.data import DataLoader

from learnable_wavelets.model.loss import mse_loss
from learnable_wavelets.module import WaveletModule


class Train:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        module: WaveletModule,
        optimizer_factory: Callable[[WaveletModule], torch.optim.Optimizer],
        device: str,
        delta: float = 1e-5,
        patience: int = 20,
        max_epochs: int = 1500,
        log_train: Callable[[int, float], None] = lambda epoch, loss: None,
        log_validation: Callable[
            [int, torch.Tensor, torch.Tensor], None
        ] = lambda epoch, x_rec, x: None,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.delta = delta
        self.patience = patience
        self.max_epochs = max_epochs

        self.log_train = log_train
        self.log_validation = log_validation

        self.epoch = 0
        self.step = 0

        self.device = device
        self.module = torch.compile(module.to(device=device), mode="max-autotune")
        self.no_progress_epochs = 0
        self.last_val_loss = None
        self.last_val_step = 0
        self.stopped = False

        self.optimizer = optimizer_factory(self.module.parameters())

    def train_step(self, x):
        self.module.train()
        self.optimizer.zero_grad()

        x_rec = self.module(x)
        loss = mse_loss(x_rec, x)

        loss.backward()
        self.optimizer.step()

        self.log_train(self.epoch, self.step, loss.item())

    def validation_step(self, x):
        self.module.eval()

        with torch.no_grad():
            x_rec = self.module(x)
            self.log_validation(self.epoch, self.step, x_rec, x)
            return mse_loss(x_rec, x).item()

    def run_epoch(self):
        for batch in self.train_loader:
            self.train_step(batch.to(device=self.device))
            self.step += 1
            if self.stopped:
                break

    def validate(self):
        if self.last_val_step == self.step:
            return
        self.last_val_step = self.step

        val_iter = iter(self.val_loader)
        batch = next(val_iter)
        loss = self.validation_step(batch.to(device=self.device))

        try:
            next(val_iter)
            raise ValueError(
                "Validation loader has more than one batch, which is not supported"
            )
        except StopIteration:
            pass

        if self.last_val_loss is None:
            self.last_val_loss = loss
            return

        if self.last_val_loss - loss < self.delta:
            self.no_progress_epochs += 1

        if self.no_progress_epochs >= self.patience:
            print(
                f"Early stopping at {self.epoch}/{self.step} with validation loss {loss}"
            )
            self.stopped = True

        self.last_val_loss = loss

    def run(self):
        for epoch in range(self.max_epochs):
            self.epoch = epoch

            self.run_epoch()
            self.validate()

            if self.stopped:
                break
