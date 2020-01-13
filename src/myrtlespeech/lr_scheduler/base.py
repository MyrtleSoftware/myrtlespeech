import copy
from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class LRSchedulerBase(_LRScheduler):
    """Provides lr schedule at ``scheduler_step_freq`` with optional warmup.

    This enables the calling of :py:meth:`scheduler.step()` after every batch
    **for all schedulers** including those that only update lrs after each
    epoch.

    Warmup, if present, is linear and follows `On the adequacy of untuned
    warmup for adaptive optimization <https://arxiv.org/abs/1910.04209>`_.

    Args:
        scheduler: An existing
            :py:class:`torch.optim.lr_scheduler._LRScheduler` learning-rate
            scheduler. If the desired schedule is constant after the
            (optional) warmup period, the :py:class:`ConstantLR` should be
            used.

        scheduler_step_freq: The frequency at which ``scheduler`` steps are
            taken.

        num_warmup_steps: The number of warmup steps. If None, no warmup is
            perfomed.

    Raises:
        :py:class:`ValueError`: if ``num_warmup_steps < 1``.

    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scheduler_step_freq: int,
        num_warmup_steps: Optional[int] = None,
    ):
        if num_warmup_steps is not None and num_warmup_steps < 1:
            raise ValueError("num_warmup_steps must be > 0.")

        self._scheduler = scheduler
        self.optimizer = self._scheduler.optimizer
        self.step_freq = scheduler_step_freq
        self.num_warmup_steps = num_warmup_steps
        self.step(self._scheduler.last_epoch)

    def step(self, epoch=None):
        """Performs a step of the lr scheduler.

        This is a no-op if ``optimizer.step()`` has not been called since this
        uses the ``optimizer._step_count`` variable.
        """
        optim_step = self.optimizer._step_count
        if optim_step % self.step_freq == 0 and optim_step != 0:
            self._scheduler.step()
        self._step_count = optim_step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def get_lr(self):
        """Returns the lr schedule.

        The existing schedule is applied before the (optional) warmup schedule.
        """
        optim_step = self.optimizer._step_count
        lrs = self._scheduler.get_lr()
        if self.num_warmup_steps is not None:
            lrs = [lr * self._warmup(optim_step) for lr in lrs]
        return lrs

    def _warmup(self, step):
        return min(1, (step / self.num_warmup_steps))

    def state_dict(self):
        """Returns the state of the scheduler as a :py:class:`dict`.

        It contains an entry for every variable in :py:meth:`self.__dict__`
        which is not the optimizer.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_scheduler")
        }
        state_dict["_scheduler"] = self._scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict: Scheduler state. Should be an object returned
                from a call to :py:meth:`state_dict`.
        """
        state_dict = copy.deepcopy(state_dict)
        _scheduler_dict = state_dict.pop("_scheduler")
        self.__dict__.update(state_dict)
        self._scheduler.__dict__.update(_scheduler_dict)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class _LambdaLR(LambdaLR):
    """Overrides :py:meth:`state_dict` method to remove ``lr_lambdas``."""

    def state_dict(self):
        state_d = super().state_dict()
        state_d.pop("lr_lambdas")
        return state_d