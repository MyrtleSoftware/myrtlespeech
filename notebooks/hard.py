import math

import torch
import torch.nn as nn
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


class HardLSTM(nn.Module):
    def __init__(
        self, in_size, hidden_size, batch_first=False, bidirectional=False
    ):
        super(HardLSTM, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        gate_size = 4 * hidden_size

        self._all_weights = []

        for direction in range(self.num_directions):
            w_ih = nn.Parameter(torch.Tensor(gate_size, in_size))
            w_hh = nn.Parameter(torch.Tensor(gate_size, hidden_size))
            b_ih = nn.Parameter(torch.Tensor(gate_size))
            b_hh = nn.Parameter(torch.Tensor(gate_size))
            layer_params = (w_ih, w_hh, b_ih, b_hh)

            suffix = "_reverse" if direction == 1 else ""
            param_names = [
                "weight_ih_l0",
                "weight_hh_l0",
                "bias_ih_l0",
                "bias_hh_l0",
            ]
            param_names = [x + suffix for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)

            self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden=None):
        if self.batch_first:
            x = x.transpose(0, 1)

        if hidden is None:
            hidden = torch.zeros(
                self.num_directions, x.shape[1], self.hidden_size
            )
            hidden = (hidden, hidden)

        weights = self.all_weights
        hidden = list(zip(*hidden))

        all_outputs = []
        all_hidden = []
        for direction in range(self.num_directions):
            output, hy = self.recurrent(
                x,
                hidden[direction],
                weights[direction],
                reverse=(direction == 1),
            )
            all_hidden.append(hy)
            all_outputs.append(output)

        output = torch.cat(all_outputs, -1)
        hy, cy = zip(*all_hidden)
        all_hidden = (
            torch.cat(hy, 0).view(self.num_directions, *hy[0].size()),
            torch.cat(cy, 0).view(self.num_directions, *cy[0].size()),
        )

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, all_hidden

    @staticmethod
    def recurrent(x, hidden, params, reverse=False):
        output = []
        steps = range(x.size(0) - 1, -1, -1) if reverse else range(x.size(0))
        for t in steps:
            hx, cx = hidden
            hidden = HardLSTMCell(x[t], hx, cx, *params)
            output.append(hidden[0])

        if reverse:
            output.reverse()

        output = torch.cat(output, 0).view(x.size(0), *output[0].size())

        return output, hidden

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]
