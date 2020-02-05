from enum import IntEnum
from typing import Optional
from typing import Tuple
from typing import TypeVar

import torch
from torch.nn.utils.rnn import PackedSequence


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


class RNNBase(torch.nn.Module):
    """A recurrent neural network.

    See :py:class:`torch.nn.LSTM`, :py:class:`torch.nn.GRU` and
    :py:class:`torch.nn.RNN` for more information as these are used internally
    (see Attributes).

    This wrapper ensures the sequence length information is correctly used by
    the RNN (i.e. using :py:func:`torch.nn.utils.rnn.pad_packed_sequence` and
    :py:func:`torch.nn.utils.rnn.pad_packed_sequence`).

    ########
    Returns result of applying the rnn to inputs.

    All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
    if :py:func:`torch.cuda.is_available` was :py:data:`True` on
    initialisation.

    Args:
        x: A Tuple where the first element is the rnn sequence input
            which is a :py:class:`torch.Tensor` with size
            ``[seq_len, batch, in_features]`` or ``[batch, seq_len,
            in_features]`` depending on whether ``batch_first=True`` and
            the and the second element represents the length of these
            *input* sequences.

        hx: The hidden state of type RNNState.

    Returns:
        A Tuple[Tuple[outputs, lengths], RNNState]. ``outputs`` is
        a :py:class:`torch.Tensor` with size ``[seq_len, batch,
        out_features]`` or ``[batch, seq_len, out_features]`` depending on
        whether ``batch_first=True``. ``lengths`` are the corresponding
        sequence lengths which will be unchanged from the input lengths.
        ``RNNState`` is the returned hidden state.
    #########

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
        super(RNNBase, self).__init__()
        if rnn_type == RNNType.LSTM:
            rnn_cls = torch.nn.LSTM
        elif rnn_type == RNNType.GRU:
            rnn_cls = torch.nn.GRU
        elif rnn_type == RNNType.BASIC_RNN:
            rnn_cls = torch.nn.RNN
        else:
            raise ValueError(f"unknown rnn_type {rnn_type}")

        self.batch_first = batch_first

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.rnn = self._add_wrapper(self.rnn, rnn_type)

        if rnn_type == RNNType.LSTM and bias and forget_gate_bias is not None:
            for l in range(num_layers):
                ih = getattr(self.rnn, f"bias_ih_l{l}")
                ih.data[hidden_size : 2 * hidden_size] = forget_gate_bias
                hh = getattr(self.rnn, f"bias_hh_l{l}")
                hh.data[hidden_size : 2 * hidden_size] = 0.0

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.rnn = self.rnn.cuda()

    def _add_wrapper(
        self, rnn: torch.nn.Module, rnn_type: RNNType
    ) -> torch.nn.Module:
        """Adds a wrapper as a workaround for pytorch 1.4.0 ONNX export bug.

        See here: https://github.com/pytorch/pytorch/issues/32976.
        """
        if torch.__version__ == "1.4.0":
            if rnn_type == RNNType.LSTM:
                return WrapLSTM(rnn)
            elif rnn_type in [RNNType.GRU, RNNType.BASIC_RNN]:
                return WrapGRUorRNN(rnn)
            else:
                raise ValueError
        return rnn


class WrapLSTM(torch.nn.Module):
    def __init__(self, module):
        super(WrapLSTM, self).__init__()
        self.module = module

    def forward(
        self, input, hx=None,
    ):
        # type: (PackedSequence, Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]  # noqa
        return self.module(input, hx)


class WrapGRUorRNN(torch.nn.Module):
    def __init__(self, module):
        super(WrapGRUorRNN, self).__init__()
        self.module = module

    def forward(
        self, input, hx=None,
    ):
        # type: (PackedSequence, Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]  # noqa
        return self.module(input, hx)


class LSTM(RNNBase):
    """See :py:class:`RNNBase` docstrings."""

    def __init__(self, **kwargs):
        assert kwargs["rnn_type"] == RNNType.LSTM
        super(LSTM, self).__init__(**kwargs)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ]:

        inp, lengths = x
        if self.use_cuda:
            inp = inp.cuda()
            if hx is not None:
                hx = hx[0].cuda(), hx[1].cuda()

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

        out, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=self.batch_first,
            total_length=total_length,
        )

        return (out, lengths), hid


class GRU_RNN(RNNBase):
    """See :py:class:`RNNBase` docstrings."""

    def __init__(self, **kwargs):
        assert kwargs["rnn_type"] in [RNNType.GRU, RNNType.BASIC_RNN]
        super(GRU_RNN, self).__init__(**kwargs)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        inp, lengths = x
        if self.use_cuda:
            inp = inp.cuda()
            if hx is not None:
                hx = hx.cuda()

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

        out, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=self.batch_first,
            total_length=total_length,
        )

        return (out, lengths), hid


def RNN(**kwargs) -> RNNBase:
    """Returns an initialized rnn as described in :py:class:`RNNBase`.

    Note that this function follows the same API as RNN and is necessary as
    :py:class:`torch.nn.LSTM` has a different hidden state type to
    :py:class:`torch.nn.GRU` and :py:class:`torch.nn.RNN`.

    Args:
        See :py:class:`RNNBase`.

    Returns:
        An initialized :py:class:`RNNBase`.
    """
    if kwargs["rnn_type"] == RNNType.LSTM:
        rnn = LSTM(**kwargs)
    elif kwargs["rnn_type"] in [RNNType.GRU, RNNType.BASIC_RNN]:
        rnn = GRU_RNN(**kwargs)
    return rnn
