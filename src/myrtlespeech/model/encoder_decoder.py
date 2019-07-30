from typing import Optional, Tuple, Union

import torch


class EncoderDecoder(torch.nn.Module):
    """An encoder-decoder sequence-to-sequence model.

    All ``encoder`` and ``decoder`` parameters and buffers are moved to the GPU
    with :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    .. todo::

        Document this

    Args:
        encoder:
        decoder:
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        .. todo::

            Document

        Args:
            x:

        """
        if self.use_cuda:
            x = x.cuda()
            if seq_lens is not None:
                seq_lens = seq_lens.cuda()

        if seq_lens is not None:
            h, seq_lens = self.encoder(x, seq_lens)
            return self.decoder(h, seq_lens)

        h = self.encoder(x)
        return self.decoder(h)
