import math
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.functional import F


@torch.jit.script
def hard_sigmoid(x):
    x = 0.2 * x + 0.5
    x = F.threshold(-x, -1.0, -1.0)
    return F.threshold(-x, 0.0, 0.0)


@torch.jit.script
def HardLSTMCell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(x, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = hard_sigmoid(ingate)
    forgetgate = hard_sigmoid(forgetgate)
    cellgate = F.hardtanh(cellgate)
    outgate = hard_sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.hardtanh(cy)

    return hy, cy


def HardLSTM(*args, **kwargs):
    bidirectional = kwargs["bidirectional"]
    if bidirectional:
        return HardLSTM2(*args, **kwargs)
    else:
        return HardLSTM1(*args, **kwargs)


class HardLSTM2(nn.Module):
    def __init__(
        self, in_size, hidden_size, batch_first=False, bidirectional=False
    ):
        assert bidirectional is True
        super(HardLSTM2, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        gate_size = 4 * hidden_size

        self.weight_ih_l0 = nn.Parameter(torch.Tensor(gate_size, in_size))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(gate_size))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(gate_size))

        # reverse
        self.weight_ih_l0_reverse = nn.Parameter(
            torch.Tensor(gate_size, in_size)
        )
        self.weight_hh_l0_reverse = nn.Parameter(
            torch.Tensor(gate_size, hidden_size)
        )
        self.bias_ih_l0_reverse = nn.Parameter(torch.Tensor(gate_size))
        self.bias_hh_l0_reverse = nn.Parameter(torch.Tensor(gate_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)

        weights_fwd = [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        ]
        if self.bidirectional:
            weights_bwd = [
                self.weight_ih_l0_reverse,
                self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse,
                self.bias_hh_l0_reverse,
            ]
            weights = [weights_fwd, weights_bwd]
        else:
            weights = [weights_fwd, weights_fwd]

        # hidden = list(zip(*hidden)) -> can't be scripted
        h0, c0 = hx
        hx_f = h0[0], c0[0]
        if self.bidirectional:
            hx_b = h0[1], c0[1]
            hidden = [hx_f, hx_b]
        else:
            hidden = [
                hx_f,
                hx_f,
            ]  # second element will be ignored. This preserves type

        all_outputs = torch.empty(0)
        hy = torch.empty(0)
        cy = torch.empty(0)
        for direction in range(self.num_directions):
            output, (h, c) = recurrent(
                x,
                hidden[direction],
                weights[direction],
                reverse=(direction == 1),
            )
            all_outputs = torch.cat([all_outputs, output], -1)
            hy = torch.cat([hy, h.unsqueeze(0)], 0)
            cy = torch.cat([cy, c.unsqueeze(0)], 0)

        if self.batch_first:
            all_outputs = all_outputs.transpose(0, 1)

        return all_outputs, (hy, cy)


class HardLSTM1(nn.Module):
    def __init__(
        self, in_size, hidden_size, batch_first=False, bidirectional=False
    ):
        assert bidirectional is False
        super(HardLSTM1, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        gate_size = 4 * hidden_size

        self.weight_ih_l0 = nn.Parameter(torch.Tensor(gate_size, in_size))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(gate_size))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(gate_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self, x: Tensor, hx: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)

        weights = [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        ]

        # hidden = list(zip(*hidden)) -> can't be scripted
        h0, c0 = hx
        hidden = h0[0], c0[0]

        output, (hy, cy) = recurrent(x, hidden, weights, reverse=False,)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (hy.unsqueeze(0), cy.unsqueeze(0))


def recurrent(
    x: Tensor,
    hidden: Tuple[Tensor, Tensor],
    params: List[Tensor],
    reverse: bool = False,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    output = torch.empty(0)
    hx, cx = hidden
    w_ih, w_hh, b_ih, b_hh = params
    if reverse:
        for t in range(x.size(0) - 1, -1, -1):
            hx, cx = HardLSTMCell(x[t], hx, cx, w_ih, w_hh, b_ih, b_hh)
            output = torch.cat([output, hx.unsqueeze(0)], 0)
        torch.flip(output, (0,))
    else:
        for t in range(x.size(0)):
            hx, cx = HardLSTMCell(x[t], hx, cx, w_ih, w_hh, b_ih, b_hh)
            output = torch.cat([output, hx.unsqueeze(0)], 0)

    return output, (hx, cx)


if __name__ == "__main__":
    in_size = 2
    hidden = 4
    seq_len = 3
    num_layers = 1
    bidirectional = True
    batch = 3
    lstm = HardLSTM(
        in_size=in_size,
        hidden_size=hidden,
        batch_first=False,
        bidirectional=bidirectional,
    )

    x = torch.randn(seq_len, batch, in_size)
    num_directions = 2 if bidirectional else 1
    zeros = torch.zeros(
        num_layers * num_directions, batch, hidden, dtype=x.dtype,
    )
    args = (x, (zeros, zeros))

    outputs = lstm(*args)
