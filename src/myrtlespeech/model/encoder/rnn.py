import torch


class RNNEncoder(torch.nn.Module):
    """

    .. todo::

        Document this!

    """

    def __init__(self, cnn, rnn):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        .. todo::

            Document


        Args: (seq_len, batch, in_features)?

        Returns: (seq_len, batch, out_features)?

        """

        # CNN wants (batch, channels, height, width)

        # LSTM wants (seq_len, batch, input_size) OR (batch, seq_len, features)
        pass
