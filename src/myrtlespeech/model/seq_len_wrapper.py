from typing import Any
from typing import Callable
from typing import Tuple

import torch


class SeqLenWrapper(torch.nn.Module):
    """Adds sequence length support to a :py:class:`torch.nn.Module`.

    Args:
        module: A :py:class:`torch.nn.Module`.
        seq_lens_fn: A Callable that takes a :py:class:`torch.Tensor`
            containing the sequence length of each element in the input ``x``
            and returns a :py:class:`torch.Tensor` giving the sequence length
            of each element in the output after ``module`` is applied.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        seq_lens_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.module = module
        self.seq_lens_fn = seq_lens_fn

    def forward(
        self, x: Tuple[Any, torch.Tensor], *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.module(x[0], *args, **kwargs)
        return result, self.seq_lens_fn(x[1])
