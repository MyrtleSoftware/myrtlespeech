import math

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from myrtlespeech.model.rnn import RNN
from myrtlespeech.model.rnn import RNNType


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_types(draw) -> st.SearchStrategy[RNNType]:
    return draw(st.sampled_from(RNNType))


# Tests -----------------------------------------------------------------------


@given(
    rnn_type=rnn_types(),
    input_size=st.integers(min_value=1, max_value=128),
    hidden_size=st.integers(min_value=1, max_value=128),
    num_layers=st.integers(min_value=1, max_value=4),
    bias=st.booleans(),
    dropout=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    bidirectional=st.booleans(),
    forget_gate_bias=st.one_of(
        st.none(), st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)
    ),
    batch_first=st.booleans(),
)
@settings(deadline=4000)
def test_correct_rnn_type_and_size_returned(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bias: int,
    dropout: float,
    bidirectional: bool,
    forget_gate_bias: float,
    batch_first: bool,
) -> None:
    """Ensures correct ``rnn`` type and initialisation."""
    rnn = RNN(
        rnn_type=rnn_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        # warning raised when num_layers > 1 and dropout != 0.0 as this option
        # set does not make sense
        dropout=0.0 if num_layers == 1 else dropout,
        bidirectional=bidirectional,
        forget_gate_bias=forget_gate_bias,
        batch_first=batch_first,
    )

    if rnn_type == RNNType.LSTM:
        assert isinstance(rnn.rnn, torch.nn.LSTM)
    elif rnn_type == RNNType.GRU:
        assert isinstance(rnn.rnn, torch.nn.GRU)
    elif rnn_type == RNNType.BASIC_RNN:
        assert isinstance(rnn.rnn, torch.nn.RNN)
    else:
        raise ValueError(f"rnn_type {rnn_type} not supported by test")

    assert rnn.rnn.input_size == input_size
    assert rnn.rnn.hidden_size == hidden_size
    assert rnn.rnn.num_layers == num_layers
    assert rnn.rnn.bias == bias
    assert rnn.rnn.batch_first == batch_first
    if num_layers > 1:
        assert math.isclose(rnn.rnn.dropout, dropout)
    assert rnn.rnn.bidirectional == bidirectional

    if not (
        rnn_type == RNNType.LSTM and bias and forget_gate_bias is not None
    ):
        return

    for l in range(num_layers):
        bias = getattr(rnn.rnn, f"bias_ih_l{l}")[hidden_size : 2 * hidden_size]
        bias += getattr(rnn.rnn, f"bias_hh_l{l}")[
            hidden_size : 2 * hidden_size
        ]
        bias = bias.cpu()
        assert torch.allclose(bias, torch.tensor(forget_gate_bias))


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
