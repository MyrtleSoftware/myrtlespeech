from typing import Optional
from typing import Tuple
from typing import Union

import torch


class FullyConnected(torch.nn.Module):
    r"""A fully connected neural network.

    All parameters and buffers are moved to the GPU with
    :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    Args:
        in_features: Size of each input sample.

        out_features: Size of each output sample.

        num_hidden_layers: Number of hidden layers. Must be a non-negative
            integer. ``0`` means no hidden layers making this module the same
            as a single :py:class:`torch.nn.Linear` layer.

        hidden_size: The number of features output by each hidden layer, if
            any.

        hidden_activation_fn: The activation function applied after each hidden
            layer, if any.

        dropout: The dropout probability to be applied between hidden layers. A
            float in [0., 1.]. Defualts to 0.

    Attributes:
        fully_connected: A :py:class:`torch.nn.Module` that implements the
            network specified by the class arguments. It is be an instance of
            :py:class:`torch.nn.Linear` if ``num_hidden_layers == 0`` otherwise
            it is an instance of :py:class:`torch.nn.Sequential`.

        in_features: See Args.

        out_features: See Args.

        dropout: See Args.

    Raises:
        :py:class:`ValueError`: If ``num_hidden_layers < 0``.

        :py:class:`ValueError`: If ``dropout is < 0 or > 1``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and hidden_size is
            not None``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and
            hidden_activation_fn is not None``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and dropout
            != 0.0``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
        dropout: float = 0,
    ):
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be >= 0 and <= 1")

        if num_hidden_layers == 0:
            if hidden_size is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_size is not None"
                )
            if hidden_activation_fn is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_activation_fn is not None"
                )
            if dropout > 1e-8:
                raise ValueError("num_hidden_layers==0 but dropout!=0.")

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fully_connected = self._build_fully_connected(
            in_features,
            out_features,
            num_hidden_layers,
            hidden_size,
            hidden_activation_fn,
            dropout,
        )

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.fully_connected = self.fully_connected.cuda()

    def _build_fully_connected(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
        dropout: float = 0,
    ) -> Union[torch.nn.Linear, torch.nn.Sequential]:
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(torch.nn.Linear(in_features, hidden_size))
            if hidden_activation_fn:
                hidden_layers.append(hidden_activation_fn)
            if dropout > 1e-8:
                hidden_layers.append(torch.nn.Dropout(p=dropout))
            assert hidden_size is not None
            in_features = hidden_size

        module = torch.nn.Linear(in_features, out_features)

        if hidden_layers:
            module = torch.nn.Sequential(*hidden_layers, module)

        return module

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
        x_inp, x_len = x

        if self.use_cuda:
            x_inp = x_inp.cuda()
            x_len = x_len.cuda()

        result = self.fully_connected(x_inp)

        del x_inp, x

        return result, x_len
