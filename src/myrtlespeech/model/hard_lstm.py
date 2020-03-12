import math
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNNType
from torch import Tensor
from typing_extensions import Final


class HardLSTM(torch.nn.Module):
    """A Hard LSTM.

    This class may be used as a drop-in replacement for a :py:class:`RNN` of
    ``rnn_type = RNNType.LSTM``. It has the following differences:
    1. In :py:class:`HardLSTM`, the sigmoid is replaced with a hard sigmoid.
    2. In :py:class:`HardLSTM`, the tanh is replaced with a hard tanh.
    3. :py:class:`HardLSTM` does not use
    :py:func:`torch.nn.utils.rnn.pad_packed_sequence`.
    4. :py:class:`HardLSTM` is implemented by hand instead of using a single
    CUDNN kernel as in :py:class:`torch.nn.LSTM`. In order to recover some
    performance, the :py:class:`HardLSTM` is :py:class:`torch.jit.script`-ed.
    5. As this class is scripted, its :py:meth:`forward` Args must be
    monomorphic and therefore cannot be of types ``RNNData, RNNState`` or
    ``Lengths``. This in turn means that :py:class:`HardLSTM` cannot subclass
    :py:class:`RNN`.

    Args:
        See :py:class:`RNN`.

    Raises:
        :py:class:`ValueError`: If ``rnn_type != RNNType.LSTM``.
    """

    hidden_size: Final[int]
    bidirectional: Final[bool]
    batch_first: Final[bool]
    use_cuda: Final[bool]
    num_layers: Final[int]

    def __init__(
        self,
        rnn_type: RNNType,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        forget_gate_bias: Optional[float] = None,
        batch_first: bool = False,
    ):
        if rnn_type != RNNType.LSTM:
            raise ValueError("HardLSTM must have rnn_type==RNNType.LSTM.")
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.rnn = gen_hard_lstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            forget_gate_bias=forget_gate_bias,
            batch_first=batch_first,
        )

        self.use_cuda = torch.cuda.is_available()

        # Send module to GPU _before_ scripting for optimal performance
        if self.use_cuda:
            self.rnn = self.rnn.cuda()

        self.rnn = torch.jit.script(self.rnn)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        r"""Returns the result of applying the hard lstm to ``(x[0], hx)``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple where the first element is the rnn sequence input and
                the second represents the length of these *input* sequences.

            hx: The Optional lstm hidden state: a Tuple[Tensor, Tensor].

        Returns:
            A Tuple of Tuples; ``res = (a, b), (c, d)`` where
            ``a`` is the lstm sequence output, ``b`` are the lengths of
            these output sequences and ``(c, d)`` are the hidden and cell
            states of the lstm.
        """
        inp, lengths = x
        hid: Tuple[Tensor, Tensor]
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(
                self.num_layers * num_directions,
                lengths.size(0),
                self.hidden_size,
                dtype=inp.dtype,
            )
            hid = (zeros, zeros)
        else:
            hid = hx

        if self.use_cuda:
            inp = inp.cuda()
            hid = hid[0].cuda(), hid[1].cuda()

        y, hid = self.rnn(inp, hid)
        return (y, lengths), hid


def gen_hard_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bias: bool = True,
    dropout: float = 0.0,
    bidirectional: bool = False,
    forget_gate_bias: Optional[float] = None,
    batch_first: bool = False,
) -> torch.nn.Module:
    """Hard LSTM constructor.

    Args:
        See :py:class:`HardLSTM`.
    """
    assert not dropout, "Dropout for HardLSTMs is not supported."
    if bidirectional:
        layer_type = HardLSTMBidirLayer
    else:
        layer_type = HardLSTMLayer

    return StackedLSTM(
        num_layers,
        layer_type,
        input_size=input_size,
        hidden_size=hidden_size,
        forget_gate_bias=forget_gate_bias,
        batch_first=batch_first,
        bidirectional=bidirectional,
        bias=bias,
    )


class StackedLSTM(torch.nn.Module):
    """Multi-layer LSTM.

    Args:
        num_layers: See :py:class:`HardLSTM`.

        layer_type: The layer type used in the :py:class:`StackedLSTM`.

        input_size: See :py:class:`HardLSTM`.

        hidden_size: See :py:class:`HardLSTM`.

        bias: See :py:class:`HardLSTM`.

        dropout: See :py:class:`HardLSTM`.

        bidirectional: See :py:class:`HardLSTM`.

        forget_gate_bias: See :py:class:`HardLSTM`.

        batch_first: See :py:class:`HardLSTM`.
    """

    _num_layers: Final[int]
    _hidden_size: Final[int]
    _num_directions: Final[int]
    _batch_first: Final[bool]
    device: Final[torch.device]

    def __init__(
        self,
        num_layers: int,
        layer_type: torch.nn.Module,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        forget_gate_bias: Optional[float] = None,
        batch_first: bool = False,
    ):

        super(StackedLSTM, self).__init__()
        # To make model attributes available in forward method for onnx export,
        # it is necessary to define the type in advance with the Final[type]
        # constructor (see class attributes below class docstring). However,
        # after a module is scripted, any attributes marked with Final are
        # not available to user as self.atrb. Hence, for those attributes that
        # are required by the user (defined by :py:class:`RNN`) that are also
        # required in the forward pass (e.g. batch_first) we define duplicate
        # atrributes (e.g. self._batch_first and self.batch_first).
        self.input_size = input_size
        self.hidden_size = self._hidden_size = hidden_size
        self.num_layers = self._num_layers = num_layers
        num_directions = 2 if bidirectional else 1
        self.num_directions = self._num_directions = num_directions
        self.batch_first = self._batch_first = batch_first
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        first_layer_args = [
            input_size,
            hidden_size,
            forget_gate_bias,
        ]
        other_layer_args = [
            hidden_size * num_directions,
            hidden_size,
            forget_gate_bias,
        ]

        layers = [layer_type(*first_layer_args)] + [
            layer_type(*other_layer_args) for _ in range(num_layers - 1)
        ]

        self.layers = torch.nn.ModuleList(layers)

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.layers = self.layers.cuda()

        self.device = next(self.layers.parameters()).device

    def forward(
        self, input: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Returns the result of applying the hard lstm to ``(input, hx)``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            input: A :py:class:`torch.Tensor` of rnn sequence inputs of size
                ``[seq_len, batch, self.input_size]``.

            hx: The lstm hidden state: a Tuple of lstm hidden state and cell
                states where both are :py:class:`torch.Tensor`s of size
                ``[self.num_directions, batch, self.hidden_size]``.

        Returns:
            A Tuple of Tuples; ``res = a, (b, c)`` where all elements are
            :py:class:`torch.Tensor`s. ``a`` is the lstm output of size
            ``[seq_len, batch, self.num_directions * self.hidden_size]``
            and ``(c, d)`` are the final lstm hidden and cell states.
        """
        if self._batch_first:
            input = input.transpose(0, 1)

        hn, cn = hx
        batch = hn.size(1)
        req_size = (
            self._num_layers,
            self._num_directions,
            batch,
            self._hidden_size,
        )
        hn = hn.view(req_size)
        cn = cn.view(req_size)

        i = 0
        output = input

        hn_out = torch.empty(0, device=self.device)
        cn_out = torch.empty(0, device=self.device)
        for rnn_layer in self.layers:
            output, (h, c) = rnn_layer(output, (hn[i], cn[i]))
            if i == 0:
                hn_out = h.unsqueeze(0)
                cn_out = c.unsqueeze(0)
            else:
                cn_out = torch.cat([cn_out, c.unsqueeze(0)], 0)
                hn_out = torch.cat([hn_out, h.unsqueeze(0)], 0)
            i += 1
        req_size_out = (-1, batch, self._hidden_size)
        out_states = hn_out.view(req_size_out), cn_out.view(req_size_out)

        if self._batch_first:
            output = output.transpose(0, 1)

        return output, out_states


class HardLSTMLayer(torch.nn.Module):
    """Hard LSTM layer.

    Args:
        See :py:class:`HardLSTM`.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    device: Final[torch.device]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = HardLSTMCell(input_size, hidden_size, forget_gate_bias)

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.cell = self.cell.cuda()

        self.device = next(self.cell.parameters()).device

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Returns the result of applying the layer to ``(x, hx)``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A :py:class:`torch.Tensor` of rnn sequence inputs of size
                ``[seq_len, batch, self.input_size]``.

            hx: The lstm hidden state: a Tuple of lstm hidden state and cell
                states where both are :py:class:`torch.Tensor`s of size
                ``[1, batch, self.hidden_size]``.

        Returns:
            A Tuple of Tuples; ``res = a, (b, c)`` where all elements are
            :py:class:`torch.Tensor`s. ``a`` is the lstm output of size
            ``[seq_len, batch, self.hidden_size]`` and ``(c, d)`` are the
            final lstm hidden and cell states.
        """
        hx = hx[0].squeeze(0), hx[1].squeeze(0)
        timesteps = x.size(0)
        # We extend y on each timestep with y = torch.cat([y, ...]) so
        # must define initial y. Normally we could use y = torch.empty(0) but
        # with onnx export, the dimensions must match for **all** concated
        # tensors, even when empty. Hence define with correct shape but this
        # means that it will not be treated as empty and will have to be
        # removed from y later.
        y = torch.empty(1, x.size(1), self.hidden_size, device=self.device)
        for t in range(timesteps):
            hy, hx = self.cell(x[t], hx)
            y = torch.cat([y, hy.unsqueeze(0)], 0)

        # Remove torch.empty element
        y = y[1:]
        h, c = hx
        return y, (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMBidirLayer(torch.nn.Module):
    """A bidirectional LSTM layer.

    Args:
        See :py:class:`HardLSTM`.
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate_bias = forget_gate_bias

        args = (
            input_size,
            hidden_size,
            forget_gate_bias,
        )
        self.fwd = HardLSTMLayer(*args)
        self.bwd = HardLSTMLayer(*args)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.fwd = self.fwd.cuda()
            self.bwd = self.bwd.cuda()

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Returns the result of applying the layer to ``(x, hx)``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A :py:class:`torch.Tensor` of rnn sequence inputs of size
                ``[seq_len, batch, self.input_size]``.

            hx: The lstm hidden state: a Tuple of lstm hidden state and cell
                states where both are :py:class:`torch.Tensor`s of size
                ``[2, batch, self.hidden_size]``.

        Returns:
            A Tuple of Tuples; ``res = a, (b, c)`` where all elements are
            :py:class:`torch.Tensor`s. ``a`` is the lstm output of size
            ``[seq_len, batch, 2 * self.hidden_size]`` and ``(c, d)`` are
            the final lstm hidden and cell states.
        """

        timesteps = x.size(0)
        batch = x.size(1)

        h0, c0 = hx
        hx_f = h0[0].unsqueeze(0), c0[0].unsqueeze(0)
        hx_b = h0[1].unsqueeze(0), c0[1].unsqueeze(0)
        x_rev = torch.flip(x, (0,))
        y_f, (h_f, c_f) = self.fwd(x, hx_f)
        y_b, (h_b, c_b) = self.bwd(x_rev, hx_b)
        y_b = torch.flip(y_b, (0,))

        ys = torch.cat([y_f.unsqueeze(2), y_b.unsqueeze(2)], 2)
        ys = ys.view((timesteps, batch, 2 * self.hidden_size))
        h = torch.cat([h_f, h_b], 0)
        c = torch.cat([c_f, c_b], 0)

        return ys, (h, c)


class HardLSTMCell(torch.nn.Module):
    """A Hard LSTM cell.

    Args:
        See :py:class:`HardLSTM`.
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
    ):
        super(HardLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate_bias = forget_gate_bias
        self.weight_ih = torch.nn.Parameter(
            torch.randn(4 * hidden_size, input_size)
        )
        self.weight_hh = torch.nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

        self.hardtanh = torch.nn.Hardtanh()

        # Hard sigmoid parameters: use Tensors instead of floats to
        # preserve type during ONNX export
        self.slope = torch.tensor([0.2], dtype=torch.float32)
        self.offset = torch.tensor([0.5], dtype=torch.float32)

        if torch.cuda.is_available():
            self = self.cuda()  # sends module's registered parameters
            self.slope = self.slope.cuda()
            self.offset = self.offset.cuda()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        if self.forget_gate_bias is not None:
            self.bias_ih.data[
                self.hidden_size : 2 * self.hidden_size
            ] = self.forget_gate_bias
            self.bias_hh.data[self.hidden_size : 2 * self.hidden_size] = 0.0

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Returns the result of applying the cell to ``(x, hx)``.

        Args:
            input: A :py:class:`torch.Tensor` of cell inputs of size
                ``[batch, self.input_size]``.

            state: The lstm hidden state: a Tuple of hidden state and cell
                states where both are :py:class:`torch.Tensor`s of size
                ``[batch, self.hidden_size]``.

        Returns:
            A Tuple of Tuples; ``res = a, (b, c)`` where all elements are
            :py:class:`torch.Tensor`s. ``a`` is the cell output of size
            ``[batch, 2 * self.hidden_size]`` and ``(c, d)`` are
            the final hidden and cell states of size
            ``[batch, self.hidden_size]``.

        """
        hx, cx = state
        gates = (
            input.matmul(self.weight_ih.t())
            + self.bias_ih
            + hx.matmul(self.weight_hh.t())
            + self.bias_hh
        )

        # gates.chunk(4, 1) is breaks onnx export so index manually:
        ingate = gates[:, : self.hidden_size]
        forgetgate = gates[:, self.hidden_size : 2 * self.hidden_size]
        cellgate = gates[:, 2 * self.hidden_size : 3 * self.hidden_size]
        outgate = gates[:, 3 * self.hidden_size :]

        ingate = torch.clamp(
            self.slope * ingate + self.offset, min=0.0, max=1.0
        )
        forgetgate = torch.clamp(
            self.slope * forgetgate + self.offset, min=0.0, max=1.0
        )
        cellgate = self.hardtanh(cellgate)
        outgate = torch.clamp(
            self.slope * outgate + self.offset, min=0.0, max=1.0
        )

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.hardtanh(cy)
        return hy, (hy, cy)
