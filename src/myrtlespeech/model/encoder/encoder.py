from typing import Optional

import torch

from myrtlespeech.model.utils import Lambda


class Encoder(torch.nn.Module):
    r"""

    .. todo::

        * Document this! with examples

    """

    def __init__(
        self, cnn: Optional[torch.nn.Module], rnn: Optional[torch.nn.Module]
    ):
        super().__init__()
        self.cnn = cnn

        self.cnn_to_rnn: Optional[Lambda] = None
        if cnn is not None:
            self.cnn_to_rnn = Lambda(lambda h: conv_to_rnn_size)

        self.rnn = rnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the result of applying ``cnn`` and ``rnn`` to ``x``.

        Args:
            x: :py:class:`torch.Tensor`.

               If ``cnn is not None`` then ``x`` must have size ``[batch,
               channels, features, seq_len]`` that is valid for ``cnn``. The
               output of ``cnn`` will be changed to have size
               ``[cnn_out_seq_len, batch, cnn_out_features]``. If ``rnn is not
               None`` then this must be valid for input to ``rnn``.

               If ``cnn is None and rnn is not None`` then ``x`` must have size
               ``[seq_len, batch, features]``.

               If ``cnn is None and rnn is None`` then the :py:class:`.Encoder`
               is the identity function that returns ``x`` so size does not
               matter.

        Returns:
            The :py:class:`torch.Tensor` after applying both ``cnn`` and
            ``rnn`` if not None. If ``cnn is not None or rnn is not None`` then
            it will have size ``[out_seq_len, batch, out_features]``. Otherwise
            the returned :py:class:`torch.Tensor` is ``x``.
        """
        h = x
        if self.cnn:
            h = self.cnn(h)
            h = self.cnn_to_rnn(h)  # type: ignore
        if self.rnn:
            h = self.rnn(h)
        return h


def conv_to_rnn_size(x: torch.Tensor) -> torch.Tensor:
    r"""Returns a 3D :py:class:`torch.Tensor` given a 4D input.

    Args:
        x: :py:class:`torch.Tensor` with size ``[batch, channels, features,
            seq_len]``

    Returns:
        ``x`` but resized to ``[seq_len, batch, channels*features]``
    """
    batch, channels, features, seq_len = x.size()
    return x.view(batch, channels * features, seq_len).permute(2, 0, 1)
