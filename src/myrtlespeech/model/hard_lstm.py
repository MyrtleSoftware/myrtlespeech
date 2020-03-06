import math
from typing import List
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import init_hidden_state
from myrtlespeech.model.rnn import RNNType
from torch import Tensor


class HardLSTM(torch.nn.Module):
    """A Hard LSTM.

    This class may be used as a drop-in replacement for a :py:class:`RNN` of
    ``rnn_type = RNNType.LSTM`` with the following differences:
    1. In :py:class:`HardLSTM`, the sigmoid is replaced with a hard sigmoid.
    2. In :py:class:`HardLSTM`, the tanh is replaced with a hard tanh.
    3. :py:class:`HardLSTM` does not use
    :py:func:`torch.nn.utils.rnn.pad_packed_sequence`.
    4. :py:class:`HardLSTM` is implemented by hand instead of using a single
    CUDNN kernel as :py:class:`torch.nn.LSTM` does. In order to recover some
    of the performance, the :py:class:`HardLSTM` is
    :py:class:`torch.jit.script`-ed.
    5. As this class is scripted, its :py:meth:`forward` Args cannot be
    cannot be of types ``RNNData, RNNState`` or ``Lengths`` (as they cannot
    be polymorphic). Incidentally, this is also the reason that
    :py:class:`HardLSTM` cannot subclass :py:class:`RNN`.

    Args:
        See :py:class:`RNN`.

    Raises:
        :py:class:`ValueError`: If ``rnn_type != RNNType.LSTM``.
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
        if rnn_type != RNNType.LSTM:
            raise ValueError("HardLSTM must have rnn_type==RNNType.LSTM.")
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.batch_first = batch_first

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
        self, x: Tuple[Tensor, Tensor], hx: Tuple[Tensor, Tensor] = None
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        r"""Returns the result of applying the hard_lstm to ``(x[0], hx)``.

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
            hx = init_hidden_state(
                batch=len(lengths),
                dtype=inp.dtype,
                hidden_size=self.hidden_size,
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

        y, hid = self.rnn(inp, hx)
        return (y, lengths), hid


def gen_hard_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bias: int = True,
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
    """Multilayer LSTM.

    Args:
        See :py:class:`HardLSTM`.
    """

    def __init__(
        self,
        num_layers: int,
        layer_type: torch.nn.Module,
        input_size: int,
        hidden_size: int,
        bias: int = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        forget_gate_bias: Optional[float] = None,
        batch_first: bool = False,
    ):

        super(StackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        first_layer_args = [
            input_size,
            hidden_size,
            forget_gate_bias,
            bidirectional,
        ]
        other_layer_args = [
            hidden_size * num_directions,
            hidden_size,
            forget_gate_bias,
            bidirectional,
        ]

        layers = [layer_type(*first_layer_args)] + [
            layer_type(*other_layer_args) for _ in range(num_layers - 1)
        ]

        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, input: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Performs forward pass of multi-layer LSTM."""

        if self.batch_first:
            input = input.transpose(0, 1)

        hn, cn = hx
        req_size = self.num_layers, self.num_directions, hn.size(1), hn.size(2)
        hn = hn.view(req_size)
        cn = hn.view(req_size)

        i = 0
        output = input
        hn_out = torch.jit.annotate(List[Tensor], [])
        cn_out = torch.jit.annotate(List[Tensor], [])
        for rnn_layer in self.layers:
            output, (h, c) = rnn_layer(output, (hn[i], cn[i]))
            hn_out += [h]
            cn_out += [c]
            i += 1

        hn_out = torch.stack(hn_out)
        cn_out = torch.stack(cn_out)
        req_size_out = (-1, hn_out.size(2), hn_out.size(3))
        out_states = hn_out.view(req_size_out), cn_out.view(req_size_out)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, out_states


class HardLSTMLayer(torch.nn.Module):
    """An LSTM layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
        bidirectional: bool = False,
        reverse: bool = False,
    ):
        assert not bidirectional
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate_bias = forget_gate_bias
        self.reverse = reverse
        self.cell = HardLSTMCell(input_size, hidden_size)
        self.reset_parameters()

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

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Performs forward pass of single LSTM layer."""
        hx = hx[0][0], hx[1][0]
        xs = x.unbind(0)
        y = torch.jit.annotate(List[Tensor], [])
        if self.reverse:
            for t in range(len(xs) - 1, -1, -1):
                hy, hx = self.cell(xs[t], hx)
                y += [hy]
            y.reverse()
        else:
            for t in range(len(xs)):
                hy, hx = self.cell(xs[t], hx)
                y += [hy]
        (h, c) = hx

        return torch.stack(y), (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMBidirLayer(torch.nn.Module):
    """A Bidirectional LSTM layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
        bidirectional: bool = True,
    ):
        assert bidirectional
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
                HardLSTMLayer(*args, reverse=False),
                HardLSTMLayer(*args, reverse=True),
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
            ys += [y]
            hs += [h]
            cs += [c]
        out = torch.stack(ys, 2)
        hs = torch.stack(hs).squeeze(1)
        cs = torch.stack(cs).squeeze(1)
        out = out.view((out.size(0), out.size(1), -1))
        return out, (hs, cs)


class HardLSTMCell(torch.nn.Module):
    """A Hard LSTM cell."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
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

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.clamp(0.2 * ingate + 0.5, min=0.0, max=1.0)
        forgetgate = torch.clamp(0.2 * forgetgate + 0.5, min=0.0, max=1.0)
        cellgate = torch.nn.functional.hardtanh_(cellgate)
        outgate = torch.clamp(0.2 * outgate + 0.5, min=0.0, max=1.0)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.nn.functional.hardtanh_(cy)

        return hy, (hy, cy)
