from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class _LRSchedulerWarmup(_LRScheduler):
    """Adds linear LR warmup to existing `lr_scheduler`.

    Follows `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Args:
        scheduler: An existing
            :py:class:`torch.optim.lr_scheduler._LRScheduler` learning-rate
            scheduler. If the desired schedule is constant after the warmup
            period, the :py:class:`ConstantLR` should be used.

        scheduler_step_freq: The frequency at which `scheduler` steps are
            taken.

        num_warmup_steps: The number of warmup steps.
    """

    def __init__(self, scheduler, scheduler_step_freq, num_warmup_steps):
        self._scheduler = scheduler
        self._optimizer = self._scheduler.optimizer
        self.step_freq = scheduler_step_freq
        self._num_warmup_steps = num_warmup_steps
        self.step(self._scheduler.last_epoch)

    def step(self, epoch=None):
        """Performs a step of the scheduler.

        This is a no-op if ``optimizer.step()`` has not been called as the
        implementation uses the ``optimizer._step_count`` variable.
        """
        optim_step = self._optimizer._step_count
        if optim_step % self.step_freq == 0 and optim_step != 0:
            self._scheduler.step()

        for param_group, lr in zip(
            self._optimizer.param_groups, self.get_lr()
        ):
            param_group["lr"] = lr

    def get_lr(self):
        """Returns the scheduler lr.

        The existing schedule is applied before the warmup schedule.
        """
        optim_step = self._optimizer._step_count
        pre_warm_lrs = self._scheduler.get_lr()
        return [lr * self._warmup(optim_step) for lr in pre_warm_lrs]

    def _warmup(self, step):
        return min(1, (step / self._num_warmup_steps))


class ConstantLR(LambdaLR):
    """A constant learning rate scheduler."""

    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(
            optimizer, lr_lambda=lambda x: x, last_epoch=last_epoch
        )
