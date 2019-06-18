from typing import Callable

import torch


class EncoderDecoder(torch.nn.Module):
    """An encoder-decoder sequence-to-sequence model.

    .. todo::

        Document this

    Args:
        feature_extractor:
        encoder:
        decoder:
    """

    def __init__(
        self,
        audio_feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        self.audio_feature_extractor = audio_feature_extractor
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        .. todo::

            Document

        Args:
            x:

        """
        features = self.audio_feature_extractor(x)
        hidden_state = self.encoder(features)
        return self.decoder(hidden_state)
