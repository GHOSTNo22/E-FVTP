from collections import OrderedDict
from typing import OrderedDict, List

import torch
from rich.console import Console
from torch.utils.data import Subset, DataLoader
from .base import ClientBase
import numpy as np

class FedvesClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        super(FedvesClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.mu = 0.03

    def _train(self):
        self.model.train()
        dataloader = DataLoader(dataset=self.trainset,
                                batch_size=self.batch_size,
                                drop_last=True
                                )
        for _ in range(self.local_epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.reshape((len(y), 4))
                x = x.to(torch.float32)
                y = y.to(torch.float32)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                for w, w_g in zip(self.model.parameters(), self.trainable_global_params):
                    w.grad.data += self.mu * (w_g.data - w.data)

                self.optimizer.step()

        return (
            list(self.model.state_dict(keep_vars=True).values()),
            len(self.trainset.dataset),
        )

    def set_parameters(
        self,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        super().set_parameters(model_params)
        self.trainable_global_params = list(
            filter(lambda p: p.requires_grad, model_params.values())
        )
