from typing import Callable
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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        pre_process_steps: Sequence[Tuple[Callable, Stage]],
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.pre_process_steps = pre_process_steps

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()

    @property
    def pre_process(self) -> Callable:
        """See Args."""

        def process(x):
            for step, stage in self.pre_process_steps:
                if stage is Stage.EVAL and self.training:
                    continue
                x = step(x)
            return x

        return process
