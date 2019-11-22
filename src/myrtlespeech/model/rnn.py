from enum import IntEnum
from typing import Optional
from typing import Tuple
from typing import Union

import torch


class RNNType(IntEnum):
    LSTM = 0
    GRU = 1
    BASIC_RNN = 2


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

    def forward(
        self,
        x: Tuple[
            Union[
                torch.tensor,
                Tuple[
                    torch.Tensor,
                    Optional[
                        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                    ],
                ],
            ],
            torch.Tensor,
        ],
    ) -> Tuple[
        Tuple[
            torch.Tensor,
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        ],
        torch.Tensor,
    ]:
        r"""Returns the result of applying the rnn to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple ``(x[0], x[1])``. ``x[0]`` can take two forms: either it
                is a tuple ``x[0] = (inp, hid)`` or it is a torch tensor
                ``x[0] = inp``. ``inp`` is the network input (a
                :py:class:`torch.Tensor`) with size ``[seq_len, batch,
                in_features]``. ``hid`` is the RNN hidden state which is
                either a length 2 Tuple of :py:class:`torch.Tensor`s or
                a single :py:class:`torch.Tensor` depending on the ``RNNType``
                (see :py:class:`torch.nn` documentation for more information).

                The return type `res[0]` will be the same as the `x[0]` type so
                you should pass `hid = None` if you would like the hidden state
                returned and it is the start-of-sequence. In this case, the
                hidden state(s) will be initialised to zero in Pytorch.

                ``x[1]`` is a :py:class:`torch.Tensor` where each entry
                represents the sequence length of the corresponding network
                *input* sequence.

        Returns:
            A Tuple ``(res[0], res[1])``. ``res[0]`` will take the same form as
                ``x[0]``: either a tuple ``res[0] = (out, hid)`` or a
                :py:class:`torch.Tensor`` torch tensor
                ``res[0] = out``. ``out`` is is the result after applying the
                RNN to ``inp``. It will have size
                ``[seq_len, batch, out_features]``. ``hid`` is the
                returned RNN hidden state which is either a length 2 Tuple of
                :py:class:`torch.Tensor`s or a single :py:class:`torch.Tensor`
                depending on the ``RNNType`` (see :py:class:`torch.nn`
                documentation for more information).

                ``res[1]`` is a :py:class:`torch.Tensor` where each entry
                represents the sequence length of the corresponding network
                *output* sequence. This will be equal to ``x[1]`` as this layer
                does not change sequence length.
        """

        if isinstance(x[0], torch.Tensor):
            inp = x[0]
            hid = None
            return_tuple = False
        elif isinstance(x[0], tuple) and len(x[0]) == 2:
            inp, hid = x[0]
            return_tuple = True
        else:
            raise ValueError(
                "`x[0]` must be of form (input, hidden) or (input)."
            )

        if self.use_cuda:
            inp = inp.cuda()
            if hid is not None:
                if isinstance(hid, tuple):  # LSTM/GRU
                    hid = hid[0].cuda(), hid[1].cuda()
                elif isinstance(hid, torch.Tensor):  # Vanilla RNN
                    hid = hid.cuda()
                else:
                    raise ValueError(
                        "hid must be an instance of class in \
                        [torch.Tensor, tuple]"
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

        del inp

        if return_tuple:
            return (out, hid), lengths
        del hid
        return out, lengths
