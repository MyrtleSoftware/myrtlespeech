from typing import Optional
from typing import Tuple

import torch


class FullyConnected(torch.nn.Module):
    r"""A fully connected neural network.

    All parameters and buffers are moved to the GPU with
    :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    Args:
        in_features: Size of each input sample.

        out_features: Size of each output sample.

        num_hidden_layers: Number of hidden layers. Must be a non-negative
            integer.

        hidden_size: The number of features output by each hidden layer, if
            any.

        hidden_activation_fn: The activation function applied after each hidden
            layer, if any.

        batch_norm: If :py:data:`True`, then batch normalization is added.

    Attributes:
        fully_connected: A :py:class:`torch.nn.Module` that implements the
            network specified by the class arguments. It is an instance of
            :py:class:`torch.nn.Sequential`.

        in_features: See Args.

        out_features: See Args.

    Raises:
        :py:class:`ValueError`: If ``num_hidden_layers < 0``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and
        hidden_size > 0.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and
        hidden_activation_fn is not None``.

        :py:class:`ValueError`: If ``num_hidden_layers > 0 and
        hidden_size <= 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
        batch_norm: bool = False,
    ):
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")
        elif num_hidden_layers == 0:
            if hidden_size is not None and hidden_size > 0:
                raise ValueError("num_hidden_layers==0 but hidden_size > 0")
            if hidden_activation_fn is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_activation_fn is not None"
                )
        else:
            if hidden_size is not None and hidden_size <= 0:
                raise ValueError(
                    "hidden_size must be > 0 when num_hidden_layers > 0"
                )

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = hidden_activation_fn
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size if hidden_size is not None else 0

        hidden_layers = []
        for i in range(num_hidden_layers + 1):
            # Hidden activation is eventually added only to the hidden layers
            # before the last FullyConnected layer. The same is for the batch
            # norm layers.
            if i < num_hidden_layers:
                out_features = self.hidden_size
            else:
                out_features = self.out_features
                hidden_activation_fn = None
                batch_norm = False

            hidden_layers.append(torch.nn.Linear(in_features, out_features))
            if batch_norm:
                hidden_layers.append(torch.nn.BatchNorm1d(out_features))
            if hidden_activation_fn is not None:
                hidden_layers.append(hidden_activation_fn)

            in_features = self.hidden_size

        self.fully_connected = torch.nn.Sequential(*hidden_layers)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.fully_connected = self.fully_connected.cuda()

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying ``fully_connected`` to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch, max_seq_len,
                in_features]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying the module to ``x[0]``. It has size ``[batch, max_seq_len,
            out_features]``.  The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())

        for i in range(len(self.fully_connected)):
            if self.batch_norm and isinstance(
                self.fully_connected[i], torch.nn.BatchNorm1d
            ):
                # Collapses the first two input dimensions (batch and seq_len)
                # and gives it to a batch norm layer. Allows handling of
                # variable sequence lengths and minibatch sizes.
                t, n = x[0].size(0), x[0].size(1)
                x_norm = self.fully_connected[i](x[0].reshape(t * n, -1))
                x = (x_norm.reshape(t, n, -1), x[1])
            else:
                x = (self.fully_connected[i](x[0]), x[1])

        return x
