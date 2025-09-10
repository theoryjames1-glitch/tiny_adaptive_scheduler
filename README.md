# tiny_adaptive_scheduler

```python
#!/usr/bin/env python3
"""
Tiny Demo: AdaptiveScheduler
----------------------------
Trains a toy linear model on random data.
Shows how AdaptiveScheduler updates learning rate from loss feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from collections import deque
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import Optional, Dict
import math


# --- minimal AdaptiveScheduler (no reward, just trend + variance) ---
class AdaptiveScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, up_factor=1.05, down_factor=0.7,
                 patience=10, cooldown=20, lr_min=1e-6, lr_max=1.0, last_epoch=-1):
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.patience = patience
        self.cooldown = cooldown
        self.lr_min = lr_min
        self.lr_max = lr_max

        self._gstate = []
        for g in optimizer.param_groups:
            self._gstate.append(dict(
                lr=g["lr"], base_lr=g["lr"],
                loss_prev=None, steps_no_improve=0, cooldown=0
            ))

        # must exist before super().__init__ calls step()
        self._pending: Optional[Dict[str, float]] = None
        super().__init__(optimizer, last_epoch)

    def step_loss(self, loss):
        """Feed a scalar loss and step scheduler."""
        val = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
        self._pending = {"loss": val}
        return self.step()

    def get_lr(self):
        return [st["lr"] for st in self._gstate]

    def step(self):
        self.last_epoch += 1
        loss = None if self._pending is None else self._pending["loss"]
        self._pending = None

        for g, st in zip(self.optimizer.param_groups, self._gstate):
            if loss is not None:
                if st["loss_prev"] is not None:
                    if loss < st["loss_prev"]:  # improving
                        st["lr"] *= self.up_factor
                        st["steps_no_improve"] = 0
                    else:  # not improving
                        st["lr"] *= self.down_factor
                        st["steps_no_improve"] += 1
                st["loss_prev"] = loss

            if st["cooldown"] > 0:
                st["cooldown"] -= 1
            elif st["steps_no_improve"] >= self.patience:
                st["lr"] *= 0.5
                st["steps_no_improve"] = 0
                st["cooldown"] = self.cooldown

            st["lr"] = float(min(max(st["lr"], self.lr_min), self.lr_max))
            g["lr"] = st["lr"]

        return [st["lr"] for st in self._gstate]


# --- tiny training demo ---
def main():
    torch.manual_seed(0)

    # synthetic regression
    X = torch.randn(256, 5)
    true_w = torch.randn(5, 1)
    y = X @ true_w + 0.1 * torch.randn(256, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = nn.Linear(5, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    sched = AdaptiveScheduler(opt, up_factor=1.03, down_factor=0.8, patience=5)

    for epoch in range(3):
        for step, (xb, yb) in enumerate(loader):
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            sched.step_loss(loss)

            if step % 10 == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"Epoch {epoch} Step {step:03d} | Loss={loss.item():.4f} | LR={lr:.5f}")


if __name__ == "__main__":
    main()
```
