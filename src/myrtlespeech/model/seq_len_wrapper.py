from typing import Callable

import torch


class SeqLenWrapper(torch.nn.Module):
    """Adds ``seq_len`` kwarg support to a :py:class:`torch.nn.Module`.

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

    def forward(self, x, seq_lens=None, *args, **kwargs):
        result = self.module(x, *args, **kwargs)
        if seq_lens is None:
            return result
        return result, self.seq_lens_fn(seq_lens)
