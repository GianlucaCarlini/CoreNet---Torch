import torch
import pytorch_lightning as pl
import numpy as np
from typing import Union, Any, Optional, Callable


class Model(pl.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Callable = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super(Model, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y = batch

        pred = self.model(x)
        loss = self.loss(pred, y)

        if self.metrics is not None:
            for name, metric in self.metrics.items():
                metric = metric.to(y)
                self.log(
                    f"train_{name}",
                    metric(pred, y),
                    prog_bar=True,
                    on_step=True,
                    logger=True,
                    on_epoch=True,
                )

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, logger=True, on_epoch=True
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y = batch

        pred = self.model(x)
        loss = self.loss(pred, y)

        if self.metrics is not None:
            for name, metric in self.metrics.items():
                metric = metric.to(y)
                self.log(
                    f"val_{name}",
                    metric(pred, y),
                    prog_bar=True,
                    on_step=True,
                    logger=True,
                    on_epoch=True,
                )

        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, logger=True, on_epoch=True
        )

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        return self.optimizer
