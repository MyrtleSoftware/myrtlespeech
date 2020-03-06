from enum import IntEnum
from typing import Optional
from typing import Tuple
from typing import TypeVar

import torch


class RNNType(IntEnum):
    LSTM = 0
    GRU = 1
    BASIC_RNN = 2


#: The type of an :py:class:`RNN` hidden state.
#:
#: Depending on the :py:class:`RNN`'s :py:class:`RNNType`, the hidden state
#: will either be a length 2 Tuple of :py:class:`torch.Tensor`\s or a single
#: :py:class:`torch.Tensor` (see :py:class:`torch.nn` documentation for more
#: information).
RNNState = TypeVar("RNNState", torch.Tensor, Tuple[torch.Tensor, torch.Tensor])


#: The type of the sequence data input to a :py:class:`RNN`.
#:
#: A :py:class:`torch.Tensor`, with size ``[seq_len, batch,
#: num_feature]`` or ``[batch, seq_len, num_feature]`` depending on whether
#: ``batch_first=True``.
RNNData = TypeVar("RNNData", bound=torch.Tensor)

#: A :py:class:`torch.Tensor` representing sequence lengths.
#:
#: An object of type :py:obj:`Lengths` will always be accompanied by a sequence
#: data object where each entry of the :py:obj:`Lengths` object
#: represents the sequence length of the corresponding element in the data
#: object batch.
Lengths = TypeVar("Lengths", bound=torch.Tensor)


class RNN(torch.nn.Module):
    """A recurrent neural network.

    See :py:class:`torch.nn.LSTM`, :py:class:`torch.nn.GRU` and
    :py:class:`torch.nn.RNN` for more information as these are used internally
    (see Attributes).

    This wrapper ensures the sequence length information is correctly used by
    the RNN (i.e. using :py:func:`torch.nn.utils.rnn.pad_packed_sequence` and
    :py:func:`torch.nn.utils.rnn.pad_packed_sequence`).

    Args:
        rnn_type: The type of recurrent neural network cell to use. See
            :py:class:`RNNType` for a list of the supported types.

        input_size: The number of features in the input.

        hidden_size: The number of features in the hidden state.

        num_layers: The number of recurrent layers.

        bias: If :py:data:`False`, then the layer does not use the bias weights
            ``b_ih`` and ``b_hh``.

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

        batch_first: If :py:data:`True`, then the input and output tensors are
            provided as ``[batch, seq_len, in_features]``.

    Attributes:
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
        dropout: float = 0.0,
        bidirectional: bool = False,
        forget_gate_bias: Optional[float] = None,
        batch_first: bool = False,
    ):
        super().__init__()
        if rnn_type == RNNType.LSTM:
            rnn_cls = torch.nn.LSTM
        elif rnn_type == RNNType.GRU:
            rnn_cls = torch.nn.GRU
        elif rnn_type == RNNType.BASIC_RNN:
            rnn_cls = torch.nn.RNN
        else:
            raise ValueError(f"unknown rnn_type {rnn_type}")

        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        if rnn_type == RNNType.LSTM and bias and forget_gate_bias is not None:
            for l in range(num_layers):
                ih = getattr(self.rnn, f"bias_ih_l{l}")
                ih.data[hidden_size : 2 * hidden_size] = forget_gate_bias
                hh = getattr(self.rnn, f"bias_hh_l{l}")
                hh.data[hidden_size : 2 * hidden_size] = 0.0

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.rnn = self.rnn.cuda()

    def forward(
        self, x: Tuple[RNNData, Lengths], hx: Optional[RNNState] = None
    ) -> Tuple[Tuple[RNNData, Lengths], RNNState]:
        r"""Returns the result of applying the rnn to ``(x[0], hx)``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple[RNNData, Lengths] where the first element is the rnn
                sequence input and the second represents the length of these
                *input* sequences.

            hx: The Optional hidden RNNState.

        Returns:
            A ``res: Tuple[Tuple[RNNData, Lengths], RNNState]`` where
                ``res[0][0]`` is the rnn sequence output, ``res[0][1]`` are
                the lengths of these output sequences and ``res[1]`` is the
                hidden state of the rnn.
        """
        inp, lengths = x

        if hx is None:
            hx = init_hidden_state(
                batch=len(lengths),
                dtype=inp.dtype,
                hidden_size=self.rnn.hidden_size,
                num_layers=self.rnn.num_layers,
                bidirectional=self.bidirectional,
                rnn_type=self.rnn_type,
            )

        if self.use_cuda:
            inp = inp.cuda()
            if self.rnn_type == RNNType.LSTM:
                hx = hx[0].cuda(), hx[1].cuda()  # type: ignore
            else:
                hx = hx.cuda()  # type: ignore

        # Record sequence length to enable DataParallel
        # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = inp.size(0 if not self.batch_first else 1)
        inp = torch.nn.utils.rnn.pack_padded_sequence(
            input=inp,
            lengths=lengths,
            batch_first=self.batch_first,
            enforce_sorted=True,
        )

        out, hid = self.rnn(inp, hx=hx)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=self.batch_first,
            total_length=total_length,
        )

        return (out, lengths), hid


def init_hidden_state(
    batch: int,
    dtype: torch.dtype,
    hidden_size: int,
    num_layers: int,
    bidirectional: int,
    rnn_type: RNNType,
) -> RNNState:
    """Returns an initial hidden state (all zeros).

    This is not deferred to the :py:class:`torch.nn.RNN` class since it is
    necessary to pass the hidden state to correctly export an onnx graph.

    Args:
        batch: batch size of input.

        dtype: PyTorch type of input.

        hidden_size: See :py:class:`RNN` initialisation docstring.

        num_layers: See :py:class:`RNN` initialisation docstring.

        bidirectional: See :py:class:`RNN` initialisation docstring.

        rnn_type: See :py:class:`RNN` initialisation docstring.
    """
    num_directions = 2 if bidirectional else 1
    zeros = torch.zeros(
        num_layers * num_directions, batch, hidden_size, dtype=dtype,
    )
    if rnn_type == RNNType.LSTM:
        hx = (zeros, zeros)
    else:
        hx = zeros
    return hx
