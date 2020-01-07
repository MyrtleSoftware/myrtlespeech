from collections import OrderedDict
from typing import Dict

import torch


def check_lr_schedule_set(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    batches_per_epoch: int,
    epochs: int,
):
    results: Dict = OrderedDict()
    step = 0
    optimizer = scheduler.optimizer
    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            optimizer.zero_grad()
            lr_sched = scheduler.get_lr()
            for idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                assert lr == lr_sched[idx], f"{lr}!={lr_sched[idx]}"
            results[step] = {"epoch": epoch, "batch": batch, "lr": lr}
            optimizer.step()
            scheduler.step()
            step += 1

    return results
