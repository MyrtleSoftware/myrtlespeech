from typing import Optional

import torch


class FullyConnected(torch.nn.Module):
    r"""A fully connected neural network.

    Args:
        in_features: foo

        out_features: foo

        num_hidden_layers: foo

        hidden_size: foo

        hidden_activation_fn: foo

    Raises:
        foo: TODO
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
    ):
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        if num_hidden_layers == 0:
            if hidden_size is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_size is not None"
                )
            if hidden_activation_fn is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_activation_fn is not None"
                )

        super().__init__()

        self.fully_connected = self._build_fully_connected(
            in_features,
            out_features,
            num_hidden_layers,
            hidden_size,
            hidden_activation_fn,
        )

    def _build_fully_connected(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
    ) -> torch.nn.Module:
        """TODO"""
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(torch.nn.Linear(in_features, hidden_size))
            if hidden_activation_fn:
                hidden_layers.append(hidden_activation_fn)
            assert hidden_size is not None
            in_features = hidden_size

        out_layer = torch.nn.Linear(in_features, out_features)

        if hidden_layers:
            return torch.nn.Sequential(*hidden_layers, out_layer)
        return out_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the result of applying ``fully_connected`` to ``x``.

        Args:
            x: :py:class:`torch.Tensor` with shape ``[batch, *, in_features]``
               where ``*`` means any number of additional dimensions.


        Returns:
            A :py:class:`torch.Tensor` with shape ``[batch, *, out_features]``
            where ``*`` means any number of additional dimensions.
        """
        return self.fully_connected(x)
