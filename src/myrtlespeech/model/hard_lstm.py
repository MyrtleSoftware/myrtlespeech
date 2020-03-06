import math
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor


def HardLSTM(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bias: int = True,
    dropout: float = 0.0,
    bidirectional: bool = False,
    forget_gate_bias: Optional[float] = None,
    batch_first: bool = False,
):
    assert not dropout
    assert forget_gate_bias is None
    if bidirectional:
        layer_type = HardLSTMBidirLayer
        dirs = 2
    else:
        layer_type = HardLSTMLayer
        dirs = 1

    return StackedLSTM(
        num_layers,
        layer_type,
        first_layer_args=[input_size, hidden_size],
        other_layer_args=[hidden_size * dirs, hidden_size],
        num_directions=dirs,
        batch_first=batch_first,
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
        self, input_size, hidden_size, batch_first=False, bidirectional=False
    ):
        assert batch_first is False
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.cell = HardLSTMCell(input_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

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
        self, input_size, hidden_size, batch_first=False, bidirectional=False
    ):
        assert bidirectional is False
        super().__init__(input_size, hidden_size, batch_first, bidirectional)

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx_f = hx[0][0], hx[1][0]
        y, (h, c) = self.recurrent(x, hx_f, reverse=False)
        return y, (h.unsqueeze(0), c.unsqueeze(0))


class HardLSTMBidirLayer(LSTMLayerBase):
    def __init__(
        self, input_size, hidden_size, batch_first=False, bidirectional=True
    ):
        assert bidirectional is True
        super().__init__(input_size, hidden_size, batch_first, bidirectional)

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
        num_layers,
        layer,
        first_layer_args,
        other_layer_args,
        num_directions=1,
        batch_first=False,
    ):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.batch_first = batch_first

        layers = [layer(*first_layer_args)] + [
            layer(*other_layer_args) for _ in range(num_layers - 1)
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
