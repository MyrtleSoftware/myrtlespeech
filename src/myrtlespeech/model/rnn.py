from enum import IntEnum
from typing import Optional
from typing import Tuple

import torch


class RNNType(IntEnum):
    LSTM = 0
    GRU = 1
    BASIC_RNN = 2


class RNN(torch.nn.Module):
    """A set of layers in a recurrent neural network.

    See :py:class:`torch.nn.LSTM`, :py:class:`torch.nn.GRU` and
    :py:class:`torch.nn.RNN` for more information as these are used internally
    (see Attributes).

    This wrapper ensures the sequence length information is correctly used by
    the RNN layer (i.e. using :py:func:`torch.nn.utils.rnn.pad_packed_sequence`
    and :py:func:`torch.nn.utils.rnn.pad_packed_sequence`).
    Moreover it handles batch normalization in the recurrent layer.

    Args:
        rnn_type: The type of recurrent neural network cell to use. See
            :py:class:`RNNType` for a list of the supported types.

        input_size: The number of features in the input.

        hidden_size: The number of features in the hidden state.

        num_layers: The number of recurrent layers.

        bias: If :py:data:`False`, then the layer does not use the bias weights
            ``b_ih`` and ``b_hh``.

        batch_first: If :py:data:`True`, then the input and output tensors are
            provided as ``[batch, seq_len, in_features]``.

        dropout: If non-zero, introduces a dropout layer on the
            outputs of each LSTM layer except the last layer,
            with dropout probability equal to ``dropout``.

        bidirectional: If :py:data:`True`, becomes a bidirectional LSTM.

        forget_gate_bias: If ``rnn_type == RNNType.LSTM`` and ``bias = True``
            then the sum of forget gate bias after initialisation equals this
            value if it is not :py:data:`None`. If it is :py:data:`None` then
            the default initialisation is used.

            See `Jozefowicz et al., 2015
            <http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf>`_.

        batch_norm: If :py:data:`True`, then batch normalization is added.

    Attributes:
        rnn_cls: A :py:class:`torch.nn.Module` that represents the rnn type
            used to build the rnn layers.

        rnn: A :py:class:`torch.LSTM`, :py:class:`torch.GRU`, or
            :py:class:`torch.RNN` instance.
    """

    def __init__(
        self,
        rnn_type: RNNType,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: int = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        forget_gate_bias: Optional[float] = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        if rnn_type == RNNType.LSTM:
            self.rnn_cls = torch.nn.LSTM
        elif rnn_type == RNNType.GRU:
            self.rnn_cls = torch.nn.GRU
        elif rnn_type == RNNType.BASIC_RNN:
            self.rnn_cls = torch.nn.RNN
        else:
            raise ValueError(f"unknown rnn_type {rnn_type}")

        self.batch_first = batch_first

        rnn_layers = []
        num_directions = 2 if bidirectional else 1
        for i in range(num_layers):
            input_features = (
                hidden_size * num_directions if i > 0 else input_size
            )

            # Batch norm is eventually added only after the first layer
            # (if batch_norm == True)
            if i > 0 and batch_norm:
                rnn_layers.append(torch.nn.BatchNorm1d(input_features))

            rnn_layer = self.rnn_cls(
                input_size=input_features,
                hidden_size=hidden_size,
                bias=bias,
                batch_first=self.batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )

            if (
                rnn_type == RNNType.LSTM
                and bias
                and forget_gate_bias is not None
            ):
                ih = getattr(rnn_layer, f"bias_ih_l0")
                ih.data[hidden_size : 2 * hidden_size] = forget_gate_bias
                hh = getattr(rnn_layer, f"bias_hh_l0")
                hh.data[hidden_size : 2 * hidden_size] = 0.0
            rnn_layers.append(rnn_layer)

        self.rnn = torch.nn.Sequential(*rnn_layers)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.rnn = self.rnn.cuda()

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the rnn layers to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[seq_len, batch,
                in_features]`` or ``[batch, seq_len, in_features]`` if
                ``batch_first=True`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying the RNN layers to ``x[0]``. It must have size ``[seq_len,
            batch, out_features]`` or ``[batch, seq_len, out_features]`` if
            ``batch_first=True``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. This will be equal to ``x[1]`` as this layer does not
            currently change sequence length.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())

        for i in range(len(self.rnn)):
            if isinstance(self.rnn[i], torch.nn.BatchNorm1d):
                # Collapses the first two input dimensions (batch and seq_len)
                # and gives it to a batch norm layer. Allows handling of
                # variable sequence lengths and minibatch sizes.
                t, n = x[0].size(0), x[0].size(1)
                x_norm = self.rnn[i](x[0].view(t * n, -1))
                x = (x_norm.view(t, n, -1), x[1])
            elif isinstance(self.rnn[i], self.rnn_cls):
                # Record sequence length to enable DataParallel
                # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
                total_length = x[0].size(0 if not self.batch_first else 1)
                input = torch.nn.utils.rnn.pack_padded_sequence(
                    input=x[0],
                    lengths=x[1],
                    batch_first=self.batch_first,
                    enforce_sorted=False,
                )
                h, _ = self.rnn[i](input)
                h, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                    sequence=h,
                    batch_first=self.batch_first,
                    total_length=total_length,
                )
                x = (h, lengths)
            else:
                raise ValueError(f"unknown layer type {type(self.rnn[i])}")

        return x
