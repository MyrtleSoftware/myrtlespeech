from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from myrtlespeech.stage import Stage


class SeqToSeq(torch.nn.Module):
    """A sequence-to-sequence model.

    All ``model`` parameters and buffers are moved to the GPU with
    :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    .. todo::

        Document this

    Args:
        model:

        loss: Callable that takes [log_probs, targets, input_lengths,
            target_lengths]?

        pre_process_steps:
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

    def _run_steps(self, steps) -> Callable:
        """TODO"""

    @property
    def pre_process(self) -> Callable:
        """TODO"""

        def process(x):
            for step, stage in self.pre_process_steps:
                if stage is Stage.EVAL and self.training:
                    continue
                x = step(x)
            return x

        return process
