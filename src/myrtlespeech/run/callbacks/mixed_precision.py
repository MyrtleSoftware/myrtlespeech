import contextlib
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

import torch
from apex import amp
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.run.callbacks.callback import ModelCallback


_ToCuda = TypeVar("_ToCuda", torch.Tensor, Dict, Tuple, List)


class MixedPrecision(ModelCallback):
    r"""Enables mixed-precision training.

    See `Mixed Precision Training <https://arxiv.org/abs/1710.03740>`_.

    Args:
        seq_to_seq: A :py:class:`.SeqToSeq` model.
        opt_level: See `NVIDIA Apex <https://nvidia.github.io/apex/amp.html>`_.
    """

    def __init__(self, seq_to_seq: SeqToSeq, opt_level: str = "O1"):
        super().__init__(seq_to_seq)
        if not torch.cuda.is_available():
            raise ValueError("cuda not available")

        seq_to_seq.model, seq_to_seq.optim = amp.initialize(
            seq_to_seq.model, seq_to_seq.optim, opt_level=opt_level
        )

        # NVIDIA's Apex module uses a content manager to handle
        # scaling of the loss. Entry/exit to/from this in
        # on_backward_begin and on_batch_end is managed using
        # an ExitStack for simplicity
        self.stack = contextlib.ExitStack()

    def _to_cuda(self, x: _ToCuda) -> _ToCuda:
        if isinstance(x, torch.Tensor):
            x = x.cuda()
        elif isinstance(x, dict):
            for key, val in x:
                x[key] = self._to_cuda(val)
        elif isinstance(x, (list, tuple)):
            x_prime: List[_ToCuda] = [self._to_cuda(val) for val in x]
            x = tuple(x_prime) if isinstance(x, tuple) else x_prime
        return x

    def on_batch_begin(self, **kwargs) -> Dict:
        r"""Moves ``kwargs["last_input"] to GPU if not already.

        ``kwargs["last_input"]`` must be a :py:class:`torch.Tensor` or a
        container of :py:class:`torch.Tensor`\s. Supported containers include
        `dict`, `tuple` and `list`.
        """
        last_input = self._to_cuda(kwargs["last_input"])
        return {"last_input": last_input}

    def on_backward_begin(self, **kwargs):
        """Scales ``kwargs["last_loss"]`` to avoid over/underflow."""
        if not self.training:
            return
        return {
            "last_loss": self.stack.enter_context(
                amp.scale_loss(kwargs["last_loss"], self.model.optim)
            )
        }

    def on_backward_end(self, **kwargs) -> None:
        """'Un'scales gradients if loss scaled during ``on_backward_begin``."""
        if not self.training:
            return
        self.stack.close()
