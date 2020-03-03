import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


def hard_sigmoid(x, slope=0.2):
    """Returns a piecewise-linear approximation of the sigmoid function.
    Empirically we find a slope of 0.2 to perform best.
    Args:
        x: A `torch.Tensor`.
    """
    return torch.clamp(input=slope * x + 0.5, min=0.0, max=1.0)


def flip(x, dim):
    """Flips tensor x along dimension dim."""
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


class ParameterRename:
    """A `torch.nn.Module` mixin to rename parameter attributes.
    A `torch.nn.Module` can include this as another base class in the class
    defintion.
    The class should define a `_param_map` dictionary of `key: value` entries.
    Both the `key` and `value` parts of an entry are strings. The `value`
    string should be the name of an existing attribute that refers to a
    `torch.nn.Parameter`. The `key` string defines a new attribute that will
    refer to the *same* `torch.nn.Parameter` as `value`.
    Multiple `key` entries can map to the same `value`.
    Both `key` and `value` parts of an entry can be used to access the
    `torch.nn.Parameter` using `getattr` and `setattr`. However, only the `key`
    entries will appear in `state_dict`.
    This allows a `torch.nn.Module` to mimic another `torch.nn.Module`'s
    interface in terms of the `torch.nn.Parameter` attributes and `state_dict`
    keys whilst using a different internal implementation or naming scheme.
    Note: This class should come *before* `torch.nn.Module` when stating the
    base classes in a class definition due to MRO/super.
    Example:
    ```
    >>> class Example(ParameterRename, torch.nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.weight = torch.nn.Parameter(torch.tensor(range(5)),
    >>>                                          requires_grad=False)
    >>>     @property
    >>>     def _param_map(self):
    >>>         return {'external_name': 'weight',
    >>>                 'external_name_1': 'weight'}
    >>>
    >>> ex = Example()
    >>> print(ex.weight, ex.external_name, ex.external_name_1, sep='\n')
    Parameter containing:
    tensor([ 0,  1,  2,  3,  4])
    Parameter containing:
    tensor([ 0,  1,  2,  3,  4])
    Parameter containing:
    tensor([ 0,  1,  2,  3,  4])
    >>> ex.external_name += 1
    >>> print(ex.weight, ex.external_name, ex.external_name_1, sep='\n')
    Parameter containing:
    tensor([ 1,  2,  3,  4,  5])
    Parameter containing:
    tensor([ 1,  2,  3,  4,  5])
    Parameter containing:
    tensor([ 1,  2,  3,  4,  5])
    >>> # the state_dict only includes the keys
    >>> print('\n'.join(sorted([str(e) for e in ex.state_dict().items()])))
    ('external_name', tensor([ 1,  2,  3,  4,  5]))
    ('external_name_1', tensor([ 1,  2,  3,  4,  5]))
    >>> ex.load_state_dict(ex.state_dict())
    >>> ex.external_name_1.fill_(10)
    Parameter containing:
    tensor([ 10,  10,  10,  10,  10])
    >>> print(ex.weight, ex.external_name, ex.external_name_1, sep='\n')
    Parameter containing:
    tensor([ 10,  10,  10,  10,  10])
    Parameter containing:
    tensor([ 10,  10,  10,  10,  10])
    Parameter containing:
    tensor([ 10,  10,  10,  10,  10])
    ```
    """

    def __getattr__(self, attr):
        # _param_map test prevents infinite recursion during __init__ phase
        if "_param_map" in self.__dict__ and attr in self._param_map:
            q_name = self._param_map[attr]
            obj = self
            for name in q_name.split("."):
                obj = getattr(obj, name)
            return obj

        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        # _param_map test prevents infinite recursion during __init__ phase
        if "_param_map" in self.__dict__ and attr in self._param_map:
            q_name = self._param_map[attr]
            obj = self
            for name in q_name.split("."):
                obj = getattr(obj, name)
            setattr(obj, attr, value)
        else:
            super().__setattr__(attr, value)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        for external, internal in self._param_map.items():
            external, internal = prefix + external, prefix + internal
            if internal in state:
                state[external] = state[internal]

        for internal in [prefix + i for i in self._param_map.values()]:
            if internal in state:
                del state[internal]

        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        for external, internal in self._param_map.items():
            external, internal = prefix + external, prefix + internal
            if external in state_dict and internal not in state_dict:
                state_dict[internal] = state_dict[external]

        for external in [prefix + e for e in self._param_map]:
            if external in state_dict:
                del state_dict[external]

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def named_parameters(self, memo=None, prefix=""):
        rev_param_map = dict(
            (internal, external)
            for external, internal in self._param_map.items()
        )

        if memo is None:
            memo = set()

        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                name = rev_param_map.get(name, name)
                yield prefix + ("." if prefix else "") + name, p

        for mname, module in self.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                name = name[len(prefix) + 1 :] if prefix else name
                name = rev_param_map.get(name, name)
                name = prefix + ("." if prefix else "") + name
                yield name, p


class HardLSTM(ParameterRename, nn.Module):
    """A long short-term memory RNN with hard activation functions.
    This class should be a drop-in replacement for `torch.nn.LSTM`.
    Args:
        hard_sigmoid_slope: Slope to use for hard (piecewise linear)
            approximation of sigmoid function.
        delta: If not `None` cell state values are clamped to this range after
            each step. The initial cell state is also clamped to this range.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        hard_sigmoid_slope=0.2,
        delta=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hard_sigmoid_slope = hard_sigmoid_slope
        self.delta = delta

        if num_layers != 1:
            raise NotImplementedError("only 1 layer currently supported")

        if batch_first:
            raise NotImplementedError("batch_first not currently supported")

        if dropout != 0.0:
            raise NotImplementedError("dropout not currently supported")

        self.bias = bias
        self.bidirectional = bidirectional

        self.lstm = HardLSTMCell(
            input_size,
            hidden_size,
            bias=bias,
            hard_sigmoid_slope=hard_sigmoid_slope,
            delta=delta,
        )

        if bidirectional:
            self.lstm_reverse = HardLSTMCell(
                input_size,
                hidden_size,
                bias=bias,
                hard_sigmoid_slope=hard_sigmoid_slope,
                delta=delta,
            )

        self._param_map = self._get_param_map()

    def forward(self, input, state=None):
        if state is None:
            state = self._zero_state(
                batch_size=input.size()[1],
                dtype=input.dtype,
                device=input.device,
            )

        output, out_state = self._run_cell(self.lstm, input, state)

        if self.bidirectional:
            r_input = flip(input, dim=0)
            output_rev, state_rev = self._run_cell(
                self.lstm_reverse, r_input, state
            )
            r_output = flip(output_rev, dim=0)
            output = torch.cat([output, r_output], dim=2)
            out_state = (
                torch.cat(
                    [out_state[0][None, :, :], state_rev[0][None, :, :]], dim=0
                ),
                torch.cat(
                    [out_state[1][None, :, :], state_rev[1][None, :, :]], dim=0
                ),
            )

        return output, out_state

    def _zero_state(self, batch_size, dtype, device):
        h_0 = torch.zeros(
            batch_size,
            self.lstm.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        c_0 = torch.zeros(
            batch_size,
            self.lstm.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        return h_0, c_0

    def _run_cell(self, cell, input, state):
        output = []
        for i, input_i in enumerate(input.split(1, dim=0)):
            input_i = input_i.view(input_i.size()[1:])
            state = cell(input_i, state)
            output.append(state[0][None, :, :])

        output = torch.cat(output, dim=0)

        return output, state

    def _get_param_map(self):
        param_map = {
            "weight_ih_l0": "lstm.weight_ih",
            "weight_hh_l0": "lstm.weight_hh",
            "bias_ih_l0": "lstm.bias_ih",
            "bias_hh_l0": "lstm.bias_hh",
        }
        if self.bidirectional:
            rev_map = {
                "weight_ih_l0_reverse": "lstm_reverse.weight_ih",
                "weight_hh_l0_reverse": "lstm_reverse.weight_hh",
                "bias_ih_l0_reverse": "lstm_reverse.bias_ih",
                "bias_hh_l0_reverse": "lstm_reverse.bias_hh",
            }
            param_map.update(rev_map)
        return param_map

    def extra_repr(self):
        return "bidirectional=%r" % self.bidirectional


class HardLSTMCell(torch.nn.Module):
    """A LSTM cell with hard activation functions.
    This class is a drop-in replacement for `torch.nn.LSTMCell`.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        hard_sigmoid_slope=0.2,
        delta=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hard_sigmoid_slope = hard_sigmoid_slope
        self.delta = delta

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx

        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(
            h_0, self.weight_hh, self.bias_hh
        )

        i, f, g, o = gates.chunk(4, 2)

        i = hard_sigmoid(i, slope=self.hard_sigmoid_slope)
        f = hard_sigmoid(f, slope=self.hard_sigmoid_slope)
        g = F.hardtanh(g)
        o = hard_sigmoid(o, slope=self.hard_sigmoid_slope)

        c_1 = (f * c_0) + (i * g)
        if self.delta is not None:
            c_1 = torch.clamp(c_1, -self.delta, self.delta)

        h_1 = o * F.hardtanh(c_1)

        return h_1, c_1

    def extra_repr(self):
        s = "input_size={input_size}"
        s += ", hidden_size={hidden_size}"
        s += ", bias={bias}"
        s += ", hard_sigmoid_slope={hard_sigmoid_slope}"
        s += ", delta={delta}"
        return s.format(**self.__dict__)
