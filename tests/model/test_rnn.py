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
from myrtlespeech.model.hard_lstm import HardLSTM
from myrtlespeech.model.rnn import RNN
from myrtlespeech.model.rnn import RNNType

from tests.utils.utils import tensors

# Fixtures and Strategies -----------------------------------------------------


class RNNHidStatus(IntEnum):
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
    draw,
    return_kwargs: bool = False,
    input_size: Optional[int] = None,
    hard_lstm: Optional[bool] = None,
) -> Union[
    st.SearchStrategy[Union[RNN, HardLSTM]],
    st.SearchStrategy[Tuple[Union[RNN, HardLSTM], Dict]],
]:
    """Returns a SearchStrategy for RNN."""
    kwargs: Dict = {}
    if not hard_lstm:
        kwargs["rnn_type"] = draw(rnn_types())

    if hard_lstm is None and kwargs["rnn_type"] == RNNType.LSTM:
        hard_lstm = st.booleans()
        if hard_lstm:
            del kwargs["rnn_type"]  # this is not an arg for HardLSTM

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
    if kwargs["num_layers"] == 1 or hard_lstm:
        kwargs["dropout"] = 0.0
    else:
        kwargs["dropout"] = draw(st.floats(min_value=0.0, max_value=1.0))

    rnn_clss: torch.nn.Module
    if hard_lstm:
        rnn_clss = HardLSTM
    else:
        rnn_clss = RNN

    if not return_kwargs:
        return rnn_clss(**kwargs)
    return rnn_clss(**kwargs), kwargs


@st.composite
def rnns_and_valid_inputs(draw) -> st.SearchStrategy[Tuple]:
    """Returns a SearchStrategy + inputs + kwargs for an RNN."""

    inp = draw(tensors(min_n_dims=3, max_n_dims=3))
    max_seq_len, batch_size, input_size = inp.size()

    rnn, kwargs = draw(rnns(return_kwargs=True, input_size=input_size))
    hard_lstm = isinstance(rnn, HardLSTM)
    if kwargs["batch_first"]:
        inp = inp.transpose(1, 0)

    hidden_state_setting = draw(rnn_hidden_settings())
    if hidden_state_setting == RNNHidStatus.HID_NONE:
        hid = None
    elif hidden_state_setting == RNNHidStatus.HID_NOT_NONE:
        num_directions = 1 + int(kwargs["bidirectional"])
        hidden_size = kwargs["hidden_size"]
        num_layers = kwargs["num_layers"]
        h_0 = torch.empty(
            [num_layers * num_directions, batch_size, hidden_size],
            requires_grad=False,
        ).normal_()
        if kwargs.get("rnn_type") in [RNNType.BASIC_RNN, RNNType.GRU]:
            hid = h_0
        elif kwargs.get("rnn_type") == RNNType.LSTM or hard_lstm:
            c_0 = h_0  # i.e. same dimensions
            hid = h_0, c_0
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

    # sort lengths since we require enforce_sorted=True
    seq_lens = seq_lens.sort(descending=True)[0]

    # hidden state
    return rnn, inp, seq_lens, hid, kwargs


# Tests -----------------------------------------------------------------------


@given(rnns(return_kwargs=True, hard_lstm=False))
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


@given(rnns(return_kwargs=True, hard_lstm=True))
def test_correct_hard_lstm_type_and_size_returned(
    rnn_kwargs: Tuple[HardLSTM, Dict]
) -> None:
    """Ensures correct ``rnn`` type and initialisation."""
    rnn, kwargs = rnn_kwargs
    assert isinstance(rnn, HardLSTM)

    assert rnn.rnn.input_size == kwargs["input_size"]
    assert rnn.rnn.hidden_size == kwargs["hidden_size"]
    assert rnn.rnn.num_layers == len(rnn.rnn.layers) == kwargs["num_layers"]
    assert rnn.rnn.bidirectional == kwargs["bidirectional"]
    num_directions = 2 if kwargs["bidirectional"] else 1
    for li in range(kwargs["num_layers"]):
        layer = rnn.rnn.layers[li]
        if li == 0:
            assert layer.input_size == kwargs["input_size"]
        else:
            assert layer.input_size == kwargs["hidden_size"] * num_directions
        assert layer.hidden_size == kwargs["hidden_size"]

    assert rnn.rnn.bias == kwargs["bias"]
    assert rnn.rnn.batch_first == kwargs["batch_first"]
    if kwargs["num_layers"] > 1:
        assert math.isclose(rnn.rnn.dropout, kwargs["dropout"])

    if not (kwargs["bias"] and kwargs["forget_gate_bias"] is not None):
        return
    hidden_size = kwargs["hidden_size"]
    for l in range(kwargs["num_layers"]):
        layer = rnn.rnn.layers[l]
        if kwargs["bidirectional"]:
            cells = [layer.fwd.cell, layer.bwd.cell]
        else:
            cells = [layer.cell]
        for cell in cells:
            bias = cell.bias_ih[hidden_size : 2 * hidden_size]
            bias += cell.bias_hh[hidden_size : 2 * hidden_size]
            bias = torch.tensor(bias).cpu()
            assert torch.allclose(
                bias, torch.tensor(kwargs["forget_gate_bias"])
            )


@given(rnns_and_valid_inputs())
def test_rnn_forward_pass_correct_shapes_returned(
    rnn_kwargs_and_valid_inputs: Tuple,
) -> None:
    """Tests forward rnn pass and checks outputs."""
    rnn, inp, seq_lens, hx, kwargs = rnn_kwargs_and_valid_inputs

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
    res = rnn((inp, seq_lens), hx=hx)

    assert isinstance(res, tuple) and len(res) == 2
    (out, out_lens), hid = res

    assert isinstance(out, torch.Tensor)
    assert out.shape == exp_out_shape
    assert torch.equal(
        out_lens.int(), seq_lens
    ), "Sequence length should be unchanged in this module"

    if kwargs.get("rnn_type") in [RNNType.BASIC_RNN, RNNType.GRU]:
        h_0 = hid
        c_0 = None
    elif kwargs.get("rnn_type") in [RNNType.LSTM, None]:  # None -> HardLSTM
        h_0, c_0 = hid

    assert h_0.shape == expected_hid_size
    if c_0 is not None:
        assert c_0.shape == expected_hid_size


@given(rnn_type=st.integers(min_value=100, max_value=300))
def test_error_raised_for_unknown_rnn_type(rnn_type: int) -> None:
    """Ensures error raised when unknown RNNType used."""
    assume(rnn_type not in list(RNNType))
    with pytest.raises(ValueError):
        RNN(rnn_type=rnn_type, input_size=1, hidden_size=1)  # type: ignore
