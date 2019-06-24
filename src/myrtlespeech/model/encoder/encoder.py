from typing import Optional

import torch


class Encoder(torch.nn.Module):
    """

    .. todo::

        Document this!

    Raises:
        :py:class:`ValueError`: if both ``cnn`` and ``rnn`` are
            :py:data:`None`.
    """

    def __init__(
        self, cnn: Optional[torch.nn.Module], rnn: Optional[torch.nn.Module]
    ):
        super().__init__()
        if cnn is None and rnn is None:
            raise ValueError("one of cnn or rnn must not be None")
        self.cnn = cnn
        self.rnn = rnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the result after applying cnn and rnn.

        Args:
            x: :py:class:`torch.Tensor` with size valid for input to ``cnn`` if
                ``cnn is not None`` else valid for input to ``rnn`` if ``rnn is
                not None``.

        Returns:
            The :py:class:`torch.Tensor` after applying both cnn and rnn.
        """
        h = x
        if self.cnn:
            h = self.cnn(h)
        if self.rnn:
            h = self.rnn(h)
        return h
