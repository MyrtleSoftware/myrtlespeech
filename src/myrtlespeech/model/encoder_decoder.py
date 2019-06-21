from typing import Callable

import torch


class EncoderDecoder(torch.nn.Module):
    """An encoder-decoder sequence-to-sequence model.

    .. todo::

        Document this

    Args:
        encoder:
        decoder:
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        .. todo::

            Document

        Args:
            x:

        """
        hidden_state = self.encoder(features)
        return self.decoder(hidden_state)
