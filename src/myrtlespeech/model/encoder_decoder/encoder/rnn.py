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
    """TODO

    Args:
        rnn_type: The type of recurrent neural network cell to use.

        input_size: The number of expected features in the input.

        hidden_size: The number of features in the hidden state.

        num_layers: Number of recurrent layers.

        bias: If :py:data:`False`, then the layer does not use bias weights
            ``b_ih`` and ``b_hh``.

        dropout: If non-zero, introduces a dropout layer on the
            outputs of each LSTM layer except the last layer,
            with dropout probability equal to ``dropout``.

        bidirectional: If :py:data:`True`, becomes a bidirectional LSTM.

        forget_gate_bias: If ``rnn_type == RNNType.LSTM`` and ``bias = True``
            then the total forget gate bias is initialised to this value if it
            is not :py:data:`None`.

            See `Jozefowicz et al., 2015
            <http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf>`_.

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

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
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
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying the rnn to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: :py:class:`torch.Tensor` with shape ``[batch, seq_len,
                in_features]``.

            seq_lens: An optional argument that, if not :py:data:`None` it must
                be a :py:class:`torch.Tensor` of size ``[batch]`` where each
                entry is an integer that gives the sequence length of the
                corresponding sequence in ``x``. If not :py:data:`None`, the
                function will also return this value (layer does not change
                sequence lengths).

        Returns:
            A :py:class:`torch.Tensor` with shape ``[batch, seq_len,
            out_features]``.
        """
        if self.use_cuda:
            x = x.cuda()
            if seq_lens is not None:
                seq_lens = seq_lens.cuda()

        h, _ = self.rnn(x)

        return h, seq_lens
