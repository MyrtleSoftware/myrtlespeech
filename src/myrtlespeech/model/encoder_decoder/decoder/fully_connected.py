from typing import Optional, Tuple, Union

import torch

from myrtlespeech.model.encoder_decoder.decoder.decoder import Decoder


class FullyConnected(Decoder):
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

    Attributes:
        fully_connected (Union[:py:class:`torch.nn.Linear`, :py:class:`torch.nn.Sequential`]):
            A :py:class:`torch.nn.Module` that implements the network specified
            by the class arguments. It is be an instance of
            :py:class:`torch.nn.Linear` if ``num_hidden_layers == 0`` otherwise
            it is an instance of :py:class:`torch.nn.Sequential`.

        in_features: See Args.

        out_features: See Args.

    Raises:
        :py:class:`ValueError`: If ``num_hidden_layers < 0``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and hidden_size is not None``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and hidden_activation_fn is not None``.
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
        self.in_features = in_features
        self.out_features = out_features
        self.fully_connected = self._build_fully_connected(
            in_features,
            out_features,
            num_hidden_layers,
            hidden_size,
            hidden_activation_fn,
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
    ) -> Union[torch.nn.Linear, torch.nn.Sequential]:
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(torch.nn.Linear(in_features, hidden_size))
            if hidden_activation_fn:
                hidden_layers.append(hidden_activation_fn)
            assert hidden_size is not None
            in_features = hidden_size

        module = torch.nn.Linear(in_features, out_features)

        if hidden_layers:
            module = torch.nn.Sequential(*hidden_layers, module)

        return module

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying ``fully_connected`` to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        See :py:meth:`.Decoder.forward`.
        """
        if self.use_cuda:
            x = x.cuda()
            if seq_lens is not None:
                seq_lens = seq_lens.cuda()

        # PyTorch linear layer documentation states batch must be the first
        # dimension but applying with seq_len first and batch second achieves
        # what we want without transposes (e.g. Linear applied to each timestep
        # for each batch element)
        result = self.fully_connected(x)
        if seq_lens is not None:
            return result, seq_lens
        return result
