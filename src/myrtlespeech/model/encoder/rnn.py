import torch


class RNNEncoder(torch.nn.Module):
    """

    .. todo::

        Document this!

    """

    def __init__(self, n_conv_layers: int):
        super().__init__()
        self.front_end = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        .. todo::

            Document


        Args: (seq_len, batch, in_features)?

        Returns: (seq_len, batch, out_features)?

        """
        pass
