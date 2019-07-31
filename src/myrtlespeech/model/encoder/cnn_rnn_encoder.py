from typing import Optional, Tuple, Union

import torch

from myrtlespeech.model.encoder.encoder import Encoder, conv_to_rnn_size
from myrtlespeech.model.utils import Lambda


class CNNRNNEncoder(Encoder):
    r"""An :py:class:`.Encoder` with CNN layers followed by RNN layers.

    All ``cnn`` and ``rnn`` parameters and buffers are moved to the GPU with
    :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    Args:
        cnn: If not :py:data:`None`, then a :py:class:`torch.nn.Module`
            containing CNN layers. Must accept input of size ``[batch,
            channels, in_features, max_in_seq_len]`` and optionally a
            ``seq_lens`` kwarg.

        rnn: If not :py:data:`None`, then a :py:class:`torch.nn.Module`
            containing RNN layers. Must accept input of size ``[seq_len, batch,
            features]`` where ``seq_len`` and ``features`` may be different to
            ``in_features`` and ``max_in_seq_len`` due to downsampling in
            ``cnn``.
    """

    def __init__(
        self, cnn: Optional[torch.nn.Module], rnn: Optional[torch.nn.Module]
    ):
        super().__init__()
        self.cnn = cnn
        self.cnn_to_rnn = Lambda(lambda h: conv_to_rnn_size(h))
        self.rnn = rnn

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            if self.cnn is not None:
                self.cnn = self.cnn.cuda()
            if self.rnn is not None:
                self.rnn = self.rnn.cuda()

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying ``cnn`` and ``rnn`` to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        See :py:meth:`myrtlespeech.model.encoder.encoder.Encoder`.
        """
        if self.use_cuda:
            x = x.cuda()
            if seq_lens is not None:
                seq_lens = seq_lens.cuda()

        h = x
        if seq_lens is not None:
            if self.cnn:
                h, seq_lens = self.cnn(h, seq_lens=seq_lens)
            h = self.cnn_to_rnn(h)
            if self.rnn:
                h, seq_lens = self.rnn(h, seq_lens=seq_lens)
            return h, seq_lens

        if self.cnn:
            h = self.cnn(h)
        h = self.cnn_to_rnn(h)
        if self.rnn:
            h = self.rnn(h)
        return h
