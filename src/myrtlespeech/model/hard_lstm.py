import math
from typing import List
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import init_hidden_state
from myrtlespeech.model.rnn import RNNType
from torch import Tensor


class HardLSTM(torch.nn.Module):
    """TODO"""

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
    assert not dropout
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


class HardLSTMCell(torch.nn.Module):
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


class LSTMLayerBase(torch.nn.Module):
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

    def recurrent(
        self, x: Tensor, hx: Tuple[Tensor, Tensor], reverse: bool,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        xs = x.unbind(0)
        y = torch.jit.annotate(List[Tensor], [])

        if reverse:
            for t in range(len(xs) - 1, -1, -1):
                hy, hx = self.cell(xs[t], hx)
                y += [hy]
            y.reverse()
        else:
            for t in range(len(xs)):
                hy, hx = self.cell(xs[t], hx)
                y += [hy]
        return torch.stack(y), hx


class HardLSTMLayer(LSTMLayerBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
    ):
        super().__init__(input_size, hidden_size, forget_gate_bias)

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx_f = hx[0][0], hx[1][0]
        y, (h, c) = self.recurrent(x, hx_f, reverse=False)
        return y, (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMBidirLayer(LSTMLayerBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_gate_bias: Optional[float] = None,
    ):
        super().__init__(input_size, hidden_size, forget_gate_bias)

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        h0, c0 = hx
        hx_f = h0[0], c0[0]
        hx_b = h0[1], c0[1]

        y_f, (h_f, c_f) = self.recurrent(x, hx_f, reverse=False)
        y_b, (h_b, c_b) = self.recurrent(x, hx_b, reverse=True)
        out = torch.stack([y_f, y_b], 2)
        out = out.view((out.size(0), out.size(1), -1))
        return out, (torch.stack([h_f, h_b]), torch.stack([c_f, c_b]))


class StackedLSTM(torch.nn.Module):
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

        first_layer_args = [input_size, hidden_size, forget_gate_bias]
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
