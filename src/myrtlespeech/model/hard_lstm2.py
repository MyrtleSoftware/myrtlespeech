import math
from typing import Tuple

import torch
from torch import Tensor


def HardLSTM(*args, **kwargs):
    bidirectional = kwargs["bidirectional"]
    if bidirectional:
        return HardLSTM2(*args, **kwargs)
    else:
        return HardLSTM1(*args, **kwargs)


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


class LSTMLayer(torch.nn.Module):
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
        y = []  # type: ignore
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


class HardLSTM2(LSTMLayer):
    def __init__(
        self, input_size, hidden_size, batch_first=False, bidirectional=False
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


class HardLSTM1(LSTMLayer):
    def __init__(
        self, input_size, hidden_size, batch_first=False, bidirectional=False
    ):
        assert bidirectional is False
        super().__init__(input_size, hidden_size, batch_first, bidirectional)

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx_f = hx[0][0], hx[1][0]
        return self.recurrent(x, hx_f, reverse=False)
