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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        .. todo::

            Document

        Args:
            x:

        """
        h = self.encoder(x)
        return self.decoder(h)
