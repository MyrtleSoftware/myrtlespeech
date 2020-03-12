import math
from typing import List
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

        rnn = gen_hard_lstm(
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
        if self.use_cuda:
            rnn = rnn.cuda()

        self.rnn = torch.jit.script(rnn)

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
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(
                self.num_layers * num_directions,
                lengths.size(0),
                self.hidden_size,
                dtype=inp.dtype,
            )
            hx = torch.jit.annotate(Tuple[Tensor, Tensor], (zeros, zeros))
        if self.use_cuda:
            inp = inp.cuda()
            hx = hx[0].cuda(), hx[1].cuda()

        y, hid = self.rnn(inp, hx)
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
        layer_type = HardLSTMLayerForward

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

    num_layers: Final[int]
    input_size: Final[int]
    hidden_size: Final[int]
    num_directions: Final[int]
    batch_first: Final[bool]
    bias: Final[bool]

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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.batch_first = batch_first
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

    def forward(
        self, input: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Performs forward pass of LSTM.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda`
        if :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            input: A :py:class:`torch.Tensor` of rnn sequence inputs.

            hx: The lstm hidden state: a Tuple of hidden state and cell state.

        Returns:
            A Tuple of Tuples; ``res = a, (b, c)`` where ``a`` is the lstm
                output and ``(c, d)`` are the hidden and cell states of the
                lstm.
        """
        # hn, cn = hx
        # return input+1., (hn+1., cn+1.)
        one = torch.tensor([1], dtype=torch.float32)
        if self.batch_first:
            input = input.transpose(0, 1)

        hn, cn = hx
        req_size = self.num_layers, self.num_directions, hn.size(1), hn.size(2)
        hn = hn.view(req_size)
        cn = cn.view(req_size)

        i = 0
        output = input + one

        hn_out = torch.empty(0)
        cn_out = torch.empty(0)
        for rnn_layer in self.layers:
            output, (h, c) = rnn_layer(output, (hn[i], cn[i]))
            # h, c = (hn[i]+one, cn[i]+one)
            if i == 0:
                hn_out = h.unsqueeze(0)
                cn_out = c.unsqueeze(0)
            else:
                cn_out = torch.cat([cn_out, c.unsqueeze(0)], 0)
                hn_out = torch.cat([hn_out, h.unsqueeze(0)], 0)
            i += 1
        req_size_out = (-1, hn_out.size(2), hn_out.size(3))
        out_states = hn_out.view(req_size_out), cn_out.view(req_size_out)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, out_states


# class HardLSTMLayer(torch.nn.Module):
#     """A single-directional LSTM layer.
#
#     Args:
#         input_size: See :py:class:`HardLSTM`.
#
#         hidden_size: See :py:class:`HardLSTM`.
#
#         forget_gate_bias: See :py:class:`HardLSTM`.
#
#         reverse: A boolean. If True, this is is a reverse LSTM layer (used in
#             a bidirectional LSTM).
#     """
#
#     input_size: Final[int]
#     hidden_size: Final[int]
#     reverse: Final[bool]
#
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         forget_gate_bias: Optional[float] = None,
#         reverse: bool = False,
#     ):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.forget_gate_bias = forget_gate_bias
#         self.reverse = reverse
#         self.cell = HardLSTMCell(input_size, hidden_size)
#         # zeros = torch.zeros(2, hidden_size)
#         # args = torch.zeros(2, input_size), (zeros, zeros)
#         # self.cell = torch.jit.trace(
#         #     HardLSTMCell(input_size, hidden_size), args, check_trace=False
#         # )
#         # self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#         if self.forget_gate_bias is not None:
#             self.cell.bias_ih.data[
#                 self.hidden_size : 2 * self.hidden_size
#             ] = self.forget_gate_bias
#             self.cell.bias_hh.data[
#                 self.hidden_size : 2 * self.hidden_size
#             ] = 0.0
#
#     def forward(
#         self, x: Tensor, hx: Tuple[Tensor, Tensor]
#     ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
#         """Performs forward pass of single LSTM layer."""
#         hx = hx[0].squeeze(0), hx[1].squeeze(0)
#         timesteps = x.size(0)
#         y = torch.empty(1, x.size(1), x.size(2))
#         if self.reverse:
#             for t in range(timesteps - 1, -1, -1):
#                 eps = x[t].mean()
#                 #hy, hx = hx[0]+eps, (hx[0]+eps, hx[1] + eps)
#                 #hy, hx = self.cell(x[t], hx)
#                 hy, hx = self.cell(hx[0]+eps, (hx[0]+eps, hx[1] + eps))
#                 y = torch.cat([y, hy.unsqueeze(0)], 0)
#             y = torch.flip(y, (0,))
#         else:
#             for t in range(timesteps):
#                 eps = x[t].mean()
#                 #hy, hx = hx[0]+eps, (hx[0]+eps, hx[1] + eps)
#                 #hy, hx = self.cell(x[t], hx)
#                 hy, hx = self.cell(hx[0]+eps, (hx[0]+eps, hx[1] + eps))
#                 y = torch.cat([y, hy.unsqueeze(0)], 0)
#         (h, c) = hx
#         # must remove the torch.empty element from the start of the list:
#         y = y[1:]
#         return y, (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMLayer(torch.nn.Module):
    """A single-directional LSTM layer.

    Args:
        input_size: See :py:class:`HardLSTM`.

        hidden_size: See :py:class:`HardLSTM`.

        forget_gate_bias: See :py:class:`HardLSTM`.

        reverse: A boolean. If True, this is is a reverse LSTM layer (used in
            a bidirectional LSTM).
    """

    input_size: Final[int]
    hidden_size: Final[int]
    reverse: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
        reverse: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate_bias = forget_gate_bias
        self.reverse = reverse
        self.cell = HardLSTMCell(input_size, hidden_size)
        # zeros = torch.zeros(2, hidden_size)
        # args = torch.zeros(2, input_size), (zeros, zeros)
        # self.cell = torch.jit.trace(
        #     HardLSTMCell(input_size, hidden_size), args, check_trace=False
        # )
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        if self.forget_gate_bias is not None:
            self.cell.bias_ih.data[
                self.hidden_size : 2 * self.hidden_size
            ] = self.forget_gate_bias
            self.cell.bias_hh.data[
                self.hidden_size : 2 * self.hidden_size
            ] = 0.0


class HardLSTMLayerForward(HardLSTMLayer):
    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx = hx[0].squeeze(0), hx[1].squeeze(0)
        timesteps = x.size(0)
        y = torch.empty(1, x.size(1), x.size(2))
        t = 0
        for _ in range(timesteps):
            eps = x[t].mean()
            # hy, hx = hx[0]+eps, (hx[0]+eps, hx[1] + eps)
            # hy, hx = self.cell(x[t], hx)
            hy, hx = self.cell(x[t] + eps, (hx[0] + eps, hx[1] + eps))
            y = torch.cat([y, hy.unsqueeze(0)], 0)
            t += 1
        (h, c) = hx
        # must remove the torch.empty element from the start of the list:
        y = y[1:]
        return y, (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMLayerReverse(HardLSTMLayer):
    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Performs forward pass of single LSTM layer."""
        hx = hx[0].squeeze(0), hx[1].squeeze(0)
        timesteps = x.size(0)
        y = torch.empty(1, x.size(1), x.size(2))
        for t in range(timesteps - 1, -1, -1):
            eps = x[t].mean()
            # hy, hx = hx[0]+eps, (hx[0]+eps, hx[1] + eps)
            hy, hx = self.cell(x[t], hx)  # this fails
            hy, hx = self.cell(hx[0] + eps, (hx[0] + eps, hx[1] + eps))
            y = torch.cat([y, hy.unsqueeze(0)], 0)
        # Remove the torch.empty element from the start of the list:
        y = y[1:]
        y = torch.flip(y, (0,))
        (h, c) = hx
        return y, (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMBidirLayer(torch.nn.Module):
    """A Bidirectional LSTM layer.

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
        self.directions = torch.nn.ModuleList(
            [
                HardLSTMLayerForward(*args, reverse=False),
                HardLSTMLayerReverse(*args, reverse=True),
            ]
        )

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Performs forward pass of single bidirectional LSTM layer."""
        h0, c0 = hx
        hx_f = h0[0].unsqueeze(0), c0[0].unsqueeze(0)
        hx_b = h0[1].unsqueeze(0), c0[1].unsqueeze(0)
        states = [hx_f, hx_b]
        i = 0
        ys = torch.jit.annotate(List[Tensor], [])
        hs = torch.jit.annotate(List[Tensor], [])
        cs = torch.jit.annotate(List[Tensor], [])
        for direction in self.directions:
            y, (h, c) = direction(x, states[i])
            ys.append(y)
            hs.append(h)
            cs.append(c)
            i + 1
        out = torch.stack(ys, 2)
        hs = torch.stack(hs).squeeze(1)
        cs = torch.stack(cs).squeeze(1)
        out = out.view((out.size(0), out.size(1), -1))
        return out, (hs, cs)


class HardLSTMCell(torch.nn.Module):
    """A Hard LSTM cell.

    Args:
        See :py:class:`HardLSTM`.
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, input_size, hidden_size):
        super(HardLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(
            torch.randn(4 * hidden_size, input_size)
        )
        self.weight_hh = torch.nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

        self.hardtanh = torch.nn.Hardtanh()

        # Hard Sigmoid parameters. Use Tensors instead of floats to
        # preserve type during ONNX export
        self.slope = torch.tensor([0.2], dtype=torch.float32)
        self.offset = torch.tensor([0.5], dtype=torch.float32)

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Performs forward pass of HardLSTM cell."""
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
