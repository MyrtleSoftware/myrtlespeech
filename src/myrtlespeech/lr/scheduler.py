from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class _LRSchedulerWarmup(_LRScheduler):
    def __init__(self, scheduler, scheduler_step_freq, num_warmup_steps):
        self._scheduler = scheduler
        self._optimizer = self._scheduler.optimizer
        self.step_freq = scheduler_step_freq
        self._num_warmup_steps = num_warmup_steps
        self.step(self._scheduler.last_epoch)

    def step(self, epoch=None):
        optim_step = self._optimizer._step_count
        if optim_step % self.step_freq == 0 and optim_step != 0:
            self._scheduler.step()

        for param_group, lr in zip(
            self._optimizer.param_groups, self.get_lr()
        ):
            param_group["lr"] = lr

    def get_lr(self):
        optim_step = self._optimizer._step_count
        pre_warm_lrs = self._scheduler.get_lr()
        return [lr * self._warmup(optim_step) for lr in pre_warm_lrs]

    def _warmup(self, step):
        return min(1, (step / self._num_warmup_steps))


class ConstantLR(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(
            optimizer, lr_lambda=lambda x: x, last_epoch=last_epoch
        )
