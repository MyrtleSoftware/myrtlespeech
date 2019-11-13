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

    if not (rnn_type == RNNType.LSTM and bias and forget_gate_bias is not None):
        return

    for l in range(num_layers):
        bias = getattr(rnn.rnn, f"bias_ih_l{l}")[hidden_size : 2 * hidden_size]
        bias += getattr(rnn.rnn, f"bias_hh_l{l}")[hidden_size : 2 * hidden_size]
        bias = torch.tensor(bias).cpu()
        assert torch.allclose(bias, torch.tensor(forget_gate_bias).cpu())


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
) -> None:
    """Tests forward rnn pass with no hidden state passed"""

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

    # TODO - test_forward_pass
    # #sample T, B, F
    # if batch_first:
    #     in_shape = (B, T, F)
    # else:
    #     in_shape = (T, B, F)
    #
    # x = torch.ones(*in_shape)


@given(rnn_type=st.integers(min_value=100, max_value=300))
def test_error_raised_for_unknown_rnn_type(rnn_type: int) -> None:
    """Ensures error raised when unknown RNNType used."""
    assume(rnn_type not in list(RNNType))
    with pytest.raises(ValueError):
        RNN(rnn_type=rnn_type, input_size=1, hidden_size=1)  # type: ignore
