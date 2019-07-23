from typing import Optional, Tuple, Union

import torch


class EncoderDecoder(torch.nn.Module):
    """An encoder-decoder sequence-to-sequence model.

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

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        .. todo::

            Document

        Args:
            x:

        """
        if seq_lens is not None:
            h, seq_lens = self.encoder(x, seq_lens)
            return self.decoder(h, seq_lens)

        h = self.encoder(x)
        return self.decoder(h)
