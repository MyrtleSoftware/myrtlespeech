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
#: The :py:class:`RNN` input type is polymorphic: either it is a Tuple
#: ``(inp, hid)`` or it is of the form ``inp``, where ``inp`` is the network
#: input, a :py:class:`torch.Tensor`, with size ``[seq_len, batch,
#: in_features]`` or ``[batch, seq_len, in_features]`` depending on whether
#: ``batch_first=True`` and ``hid`` is the :py:class:`RNN` hidden state of type
#: :py:class:`RNNState`.
RNNData = TypeVar(
    "RNNData",
    torch.Tensor,
    Tuple[torch.Tensor, Optional[RNNState]],  # type: ignore
)

#: A :py:class:`torch.Tensor` representing sequence lengths.
#:
#: An object of type :py:obj:`Lengths` will always be accompanied by a sequence
#: data object where each entry of the :py:obj:`Lengths` object represents the
#: sequence length of the corresponding element in the data object batch.
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

    def forward(self, x: Tuple[RNNData, Lengths]) -> Tuple[RNNData, Lengths]:
        r"""Returns the result of applying the rnn to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple[RNNData, Lengths] where the first element is the rnn
                sequence input and the second represents the length of these
                *input* sequences.

        Returns:
            A Tuple[RNNData, Lengths] where the first element is the rnn
                sequence output and the second represents the length of these
                *output* sequences. These lengths will be unchanged from the
                input lengths.

                The sequence output of :py:obj:`RNNData` will have the same
                subtype as ``x[0]`` so, if the user would like the hidden state
                returned at the start-of-sequence, they should pass a hidden
                state of :py:data:`None` and PyTorch will initialise the
                hidden state(s) to zero.
        """

        if isinstance(x[0], torch.Tensor):
            inp = x[0]
            hid = None
            return_tuple = False
        elif isinstance(x[0], tuple) and len(x[0]) == 2:
            inp, hid = x[0]
            return_tuple = True
        else:
            raise ValueError("`x[0]` must be of type RNNData.")

        if self.use_cuda:
            inp = inp.cuda()
            if hid is not None:
                if isinstance(hid, tuple) and len(hid) == 2:  # LSTM
                    hid = hid[0].cuda(), hid[1].cuda()
                elif isinstance(hid, torch.Tensor):  # Vanilla RNN/GRU
                    hid = hid.cuda()
                else:
                    raise ValueError(
                        "hid must be a length 2 Tuple or a torch.Tensor."
                    )

        # Record sequence length to enable DataParallel
        # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = inp.size(0 if not self.batch_first else 1)
        inp = torch.nn.utils.rnn.pack_padded_sequence(
            input=inp,
            lengths=x[1],
            batch_first=self.batch_first,
            enforce_sorted=False,
        )

        out, hid = self.rnn(inp, hx=hid)

        out, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=self.batch_first,
            total_length=total_length,
        )

        if return_tuple:
            return (out, hid), lengths
        return out, lengths
