import math
from enum import IntEnum
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from myrtlespeech.model.rnn import RNN
from myrtlespeech.model.rnn import RNNType

from tests.utils.utils import tensors

# Fixtures and Strategies -----------------------------------------------------


class RNNHidStatus(IntEnum):
    NO_HID = 0
    HID_NONE = 1
    HID_NOT_NONE = 2


@st.composite
def rnn_hidden_settings(draw) -> st.SearchStrategy[RNNHidStatus]:
    return draw(st.sampled_from(RNNHidStatus))


@st.composite
def rnn_types(draw) -> st.SearchStrategy[RNNType]:
    return draw(st.sampled_from(RNNType))


@st.composite
def rnns(
    draw, return_kwargs: bool = False, input_size: Optional[int] = None
) -> Union[
    st.SearchStrategy[RNN], st.SearchStrategy[Tuple[RNN, Dict]],
]:
    """Returns a SearchStrategy for RNN."""
    kwargs = {}
    kwargs["rnn_type"] = draw(rnn_types())
    if input_size is None:
        kwargs["input_size"] = draw(st.integers(min_value=1, max_value=128))
    else:
        kwargs["input_size"] = input_size
    kwargs["hidden_size"] = draw(st.integers(min_value=1, max_value=128))
    kwargs["num_layers"] = draw(st.integers(min_value=1, max_value=4))
    kwargs["bias"] = draw(st.booleans())
    kwargs["bidirectional"] = draw(st.booleans())
    kwargs["forget_gate_bias"] = draw(
        st.one_of(st.none(), st.floats(min_value=-10.0, max_value=10.0))
    )
    kwargs["batch_first"] = draw(st.booleans())
    if kwargs["num_layers"] == 1:
        kwargs["dropout"] = 0.0
    else:
        kwargs["dropout"] = draw(st.floats(min_value=0.0, max_value=1.0))
    if not return_kwargs:
        return RNN(**kwargs)
    return RNN(**kwargs), kwargs


@st.composite
def rnns_and_valid_inputs(draw) -> st.SearchStrategy[Tuple]:
    """Returns a SearchStrategy +inputs + kwargs for an RNN."""

    inp = draw(tensors(min_n_dims=3, max_n_dims=3))
    max_seq_len, batch_size, input_size = inp.size()

    rnn, kwargs = draw(rnns(return_kwargs=True, input_size=input_size))

    if kwargs["batch_first"]:
        inp = inp.transpose(1, 0)

    hidden_state_setting = draw(rnn_hidden_settings())
    if hidden_state_setting == RNNHidStatus.NO_HID:
        data = inp
    elif hidden_state_setting == RNNHidStatus.HID_NONE:
        data = (inp, None)
    elif hidden_state_setting == RNNHidStatus.HID_NOT_NONE:
        num_directions = 1 + int(kwargs["bidirectional"])
        hidden_size = kwargs["hidden_size"]
        num_layers = kwargs["num_layers"]
        h_0 = torch.empty(
            [num_layers * num_directions, batch_size, hidden_size],
            requires_grad=False,
        ).normal_()
        if kwargs["rnn_type"] in [RNNType.BASIC_RNN, RNNType.GRU]:
            hid = h_0
        elif kwargs["rnn_type"] == RNNType.LSTM:
            c_0 = h_0  # i.e. same dimensions
            hid = h_0, c_0

        data = (inp, hid)
    else:
        raise ValueError(
            f"hidden_state_setting == {RNNHidStatus.HID_NOT_NONE} "
            f"not recognized."
        )

    seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )

    # hidden state
    return rnn, data, seq_lens, kwargs


# Tests -----------------------------------------------------------------------


@given(rnns(return_kwargs=True))
def test_correct_rnn_type_and_size_returned(
    rnn_kwargs: Tuple[RNN, Dict]
) -> None:
    """Ensures correct ``rnn`` type and initialisation."""
    rnn, kwargs = rnn_kwargs
    if kwargs["rnn_type"] == RNNType.LSTM:
        assert isinstance(rnn.rnn, torch.nn.LSTM)
    elif kwargs["rnn_type"] == RNNType.GRU:
        assert isinstance(rnn.rnn, torch.nn.GRU)
    elif kwargs["rnn_type"] == RNNType.BASIC_RNN:
        assert isinstance(rnn.rnn, torch.nn.RNN)
    else:
        raise ValueError(
            f"rnn_type {kwargs['rnn_type']} not " f"supported by test"
        )

    assert rnn.rnn.input_size == kwargs["input_size"]
    assert rnn.rnn.hidden_size == kwargs["hidden_size"]
    assert rnn.rnn.num_layers == kwargs["num_layers"]
    assert rnn.rnn.bias == kwargs["bias"]
    assert rnn.rnn.batch_first == kwargs["batch_first"]
    if kwargs["num_layers"] > 1:
        assert math.isclose(rnn.rnn.dropout, kwargs["dropout"])
    assert rnn.rnn.bidirectional == kwargs["bidirectional"]

    if not (
        kwargs["rnn_type"] == RNNType.LSTM
        and kwargs["bias"]
        and kwargs["forget_gate_bias"] is not None
    ):
        return
    hidden_size = kwargs["hidden_size"]
    for l in range(kwargs["num_layers"]):
        bias = getattr(rnn.rnn, f"bias_ih_l{l}")[hidden_size : 2 * hidden_size]
        bias += getattr(rnn.rnn, f"bias_hh_l{l}")[
            hidden_size : 2 * hidden_size
        ]
        bias = torch.tensor(bias).cpu()
        assert torch.allclose(bias, torch.tensor(kwargs["forget_gate_bias"]))


@given(rnns_and_valid_inputs())
def test_rnn_forward_pass_correct_shapes_returned(
    rnn_kwargs_and_valid_inputs: Tuple,
) -> None:
    """Tests forward rnn pass and checks outputs."""
    rnn, data, seq_lens, kwargs = rnn_kwargs_and_valid_inputs

    # determine subtype of inp
    if isinstance(data, torch.Tensor):
        inp = data
        hidden_present_in_input = False
    elif isinstance(data, tuple):
        inp, hid_inp = data
        hidden_present_in_input = True
    else:
        raise ValueError

    # Get expected out shapes
    hidden_size = kwargs["hidden_size"]
    num_layers = kwargs["num_layers"]
    num_directions = 1 + int(kwargs["bidirectional"])
    exp_out_feat = hidden_size * num_directions
    if kwargs["batch_first"]:
        batch_size, max_seq_len, _ = inp.shape
        exp_out_shape = batch_size, max_seq_len, exp_out_feat
    else:
        max_seq_len, batch_size, _ = inp.shape
        exp_out_shape = max_seq_len, batch_size, exp_out_feat

    expected_hid_size = (num_layers * num_directions, batch_size, hidden_size)

    # check data generation
    assert batch_size == len(seq_lens)
    # Run forward pass
    res = rnn((data, seq_lens))

    assert isinstance(res, tuple) and len(res) == 2
    ret_data, out_lens = res

    if hidden_present_in_input:  # hidden state must be present in output too
        out, hid = ret_data
    else:
        out = ret_data

    assert isinstance(out, torch.Tensor)
    assert out.shape == exp_out_shape
    assert torch.equal(
        out_lens.int(), seq_lens
    ), "Sequence length should be unchanged in this module"

    if not hidden_present_in_input:
        return

    if kwargs["rnn_type"] in [RNNType.BASIC_RNN, RNNType.GRU]:
        h_0 = hid
        c_0 = None
    elif kwargs["rnn_type"] == RNNType.LSTM:
        h_0, c_0 = hid

    assert h_0.shape == expected_hid_size
    if c_0 is not None:
        assert c_0.shape == expected_hid_size


@given(
    rnn_type=rnn_types(),
    input_size=st.integers(min_value=1, max_value=8),
    hidden_size=st.integers(min_value=1, max_value=8),
    num_layers=st.integers(min_value=1, max_value=3),
    bias=st.booleans(),
    dropout=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    bidirectional=st.booleans(),
    forget_gate_bias=st.one_of(
        st.none(), st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)
    ),
    batch_first=st.booleans(),
    batch_size=st.integers(min_value=1, max_value=4),
    max_seq_len=st.integers(min_value=1, max_value=8),
)
@settings(deadline=4000)
def test_rnn_forward_pass_no_hidden(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bias: int,
    dropout: float,
    bidirectional: bool,
    forget_gate_bias: float,
    batch_first: bool,
    batch_size: int,
    max_seq_len: int,
) -> None:
    """Tests forward rnn pass with no hidden state passed"""

    rnn = RNN(
        rnn_type=rnn_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        # warning raised when num_layers > 1 and dropout != 0.0 as this option
        # set does not make sense:
        dropout=0.0 if num_layers == 1 else dropout,
        bidirectional=bidirectional,
        forget_gate_bias=forget_gate_bias,
        batch_first=batch_first,
    )
    num_directions = 1 + int(bidirectional)
    tensor = torch.empty(
        [max_seq_len, batch_size, input_size], requires_grad=False
    ).normal_()

    in_seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )
    out_shape = hidden_size * num_directions
    exp_shape = (max_seq_len, batch_size, out_shape)

    if batch_first:
        tensor = tensor.transpose(1, 0)
        exp_shape = (batch_size, max_seq_len, out_shape)
    res = rnn((tensor, in_seq_lens))

    assert isinstance(res, tuple) and len(res) == 2
    assert isinstance(res[0], torch.Tensor)
    assert res[0].shape == exp_shape
    assert torch.allclose(res[1].int(), in_seq_lens)


@given(
    rnn_type=rnn_types(),
    input_size=st.integers(min_value=1, max_value=4),
    hidden_size=st.integers(min_value=1, max_value=4),
    num_layers=st.integers(min_value=1, max_value=2),
    bias=st.booleans(),
    dropout=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    bidirectional=st.booleans(),
    forget_gate_bias=st.one_of(
        st.none(), st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)
    ),
    batch_first=st.booleans(),
    batch_size=st.integers(min_value=1, max_value=2),
    max_seq_len=st.integers(min_value=1, max_value=4),
    hidden_is_none=st.booleans(),
)
@settings(deadline=4000)
def test_rnn_forward_pass_hidden_passed(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bias: int,
    dropout: float,
    bidirectional: bool,
    forget_gate_bias: float,
    batch_first: bool,
    batch_size: int,
    max_seq_len: int,
    hidden_is_none: bool,
) -> None:
    """Tests forward rnn pass.

    Test both hidden=None and hidden=(h_0, c_0)."""

    rnn = RNN(
        rnn_type=rnn_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        # warning raised when num_layers > 1 and dropout != 0.0 as this option
        # set does not make senseno:
        dropout=0.0 if num_layers == 1 else dropout,
        bidirectional=bidirectional,
        forget_gate_bias=forget_gate_bias,
        batch_first=batch_first,
    )
    num_directions = 1 + int(bidirectional)
    tensor = torch.empty(
        [max_seq_len, batch_size, input_size], requires_grad=False
    ).normal_()

    in_seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )
    # hidden state:
    hidden = None
    if not hidden_is_none:
        h_0 = torch.empty(
            [num_layers * num_directions, batch_size, hidden_size],
            requires_grad=False,
        ).normal_()
        if rnn_type in [RNNType.BASIC_RNN, RNNType.GRU]:
            hidden = h_0
        elif rnn_type == RNNType.LSTM:
            c_0 = h_0  # i.e. same dimensions
            hidden = h_0, c_0

    out_shape = hidden_size * num_directions
    exp_shape = (max_seq_len, batch_size, out_shape)
    if batch_first:
        tensor = tensor.transpose(1, 0)
        exp_shape = (batch_size, max_seq_len, out_shape)

    res = rnn(((tensor, hidden), in_seq_lens))

    assert isinstance(res, tuple) and len(res) == 2
    assert isinstance(res[0], tuple)
    assert res[0][0].shape == exp_shape
    assert torch.allclose(res[1].int(), in_seq_lens)

    # check hidden
    hid = res[0][1]
    if rnn_type in [RNNType.BASIC_RNN, RNNType.GRU]:
        assert isinstance(hid, torch.Tensor)
    elif rnn_type == RNNType.LSTM:
        assert isinstance(hid, tuple)
        assert len(hid) == 2
        assert isinstance(hid[0], torch.Tensor) and isinstance(
            hid[1], torch.Tensor
        )
    else:
        raise TypeError(f"RNNType={rnn_type} was not expected")


@given(rnn_type=st.integers(min_value=100, max_value=300))
def test_error_raised_for_unknown_rnn_type(rnn_type: int) -> None:
    """Ensures error raised when unknown RNNType used."""
    assume(rnn_type not in list(RNNType))
    with pytest.raises(ValueError):
        RNN(rnn_type=rnn_type, input_size=1, hidden_size=1)  # type: ignore
