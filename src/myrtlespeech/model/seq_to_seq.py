from collections import OrderedDict
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
from myrtlespeech.run.stage import Stage


class SeqToSeq(torch.nn.Module):
    """A generic sequence-to-sequence model.

    Args:
        model: A :py:class:`torch.nn.Module`.

        loss: A :py:class:`torch.nn.Module` to compute the loss.

        pre_process_steps: A sequence of preprocessing steps. For each
            ``(callable, stage)`` tuple in the sequence, the ``callable``
            should accept as input the output from the previous ``callable`` in
            the sequence or raw data if it is the first. The ``callable``
            should only be applied when the current training stage matches
            ``stage``.  :py:data:`.SeqToSeq.pre_process` returns a ``Callable``
            that handles this automatically based on
            :py:class:`.SeqToSeq.training`.

        optim: An optional :py:class:`torch.optim.Optimizer` initialized with
            model parameters to update.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        pre_process_steps: Sequence[Tuple[Callable, Stage]],
        optim: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.pre_process_steps = pre_process_steps
        self.optim = optim

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()

    @property
    def pre_process(self) -> Callable:
        """See Args."""

        def process(x):
            for step, stage in self.pre_process_steps:
                if stage is Stage.TRAIN and not self.training:
                    continue
                if stage is Stage.EVAL and self.training:
                    continue
                x = step(x)
            return x

        return process

    def state_dict(self) -> OrderedDict:
        """Returns state dict."""
        state: OrderedDict = OrderedDict()
        state["model"] = self.model.state_dict()

        if self.optim is not None:
            state["optim"] = self.optim.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Mapping[str, torch.Tensor]],
        strict: bool = True,
    ):
        """See :py:meth:`~torch.nn.Module.load_state_dict`."""
        self.model.load_state_dict(state_dict["model"], strict=strict)

        if self.optim is not None:
            self.optim.load_state_dict(state_dict["optim"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            # Update the optimizer lrs to reflect correct lr_scheduler value
            for param_group, lr in zip(
                self.lr_scheduler.optimizer.param_groups,
                self.lr_scheduler.get_lr(),
            ):
                param_group["lr"] = lr
