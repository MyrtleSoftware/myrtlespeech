from typing import Callable
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

        pre_process_steps: A Tuple ``(pre_load_transforms,
            post_load_transforms)``. See Attributes for more information.

        optim: An optional :py:class:`torch.optim.Optimizer` initialized with
            model parameters to update.

    Attributes:
        pre_load_transforms: A sequence of preprocessing steps to be applied
            during loading of audio file. For each ``(callable, stage)`` Tuple
            in the sequence, the ``callable`` should accept as input the
            output from the previous ``callable`` in the sequence or a
            filepath if it is the first. ``callable`` should only be applied
            when the current training stage matches ``stage``.
            :py:data:`.SeqToSeq.pre_load_transforms` returns a ``Callable``
            that handles this automatically based on
            :py:class:`.SeqToSeq.training`.

        post_load_transforms: A sequence of preprocessing steps to be applied
            to audio :py:class`torch.Tensor`. For each ``(callable, stage)``
            Tuple in the sequence, the ``callable`` should accept as input the
            output from the previous ``callable`` in the sequence or the
            loaded audio if it is the first.
            :py:data:`.SeqToSeq.post_load_transforms` returns a ``Callable``
            that handles this automatically based on
            :py:class:`.SeqToSeq.training`.


    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        pre_process_steps: Tuple[
            Sequence[Tuple[Callable, Stage]], Sequence[Tuple[Callable, Stage]]
        ],
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

    def _get_pre_process(self, pre_process_type: str) -> Optional[Callable]:
        """See Args."""
        if pre_process_type == "post_load":
            step_and_stages = self.pre_process_steps[1]
        elif pre_process_type == "pre_load":
            step_and_stages = self.pre_process_steps[0]
        else:
            raise ValueError(
                f"pre_process_type = {pre_process_type} not " "not recognized."
            )
        if not step_and_stages:
            return None

        def process(x):
            for step, stage in step_and_stages:
                if stage is Stage.TRAIN and not self.training:
                    continue
                if stage is Stage.EVAL and self.training:
                    continue
                x = step(x)
            return x

        return process

    @property
    def pre_load_transforms(self) -> Optional[Callable]:
        return self._get_pre_process("pre_load")

    @property
    def post_load_transforms(self) -> Optional[Callable]:
        return self._get_pre_process("post_load")
