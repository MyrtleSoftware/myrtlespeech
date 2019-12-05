import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
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
    bidirectional=st.booleans(),
    forget_gate_bias=st.one_of(
        st.none(), st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)
    ),
    batch_norm=st.booleans(),
)
def test_correct_rnn_type_and_size_returned(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bias: bool,
    bidirectional: bool,
    forget_gate_bias: float,
    batch_norm: bool,
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
        dropout=0.0,
        bidirectional=bidirectional,
        forget_gate_bias=forget_gate_bias,
        batch_norm=batch_norm,
    )

    num_directions = 2 if bidirectional else 1
    for i in range(num_layers):
        input_features = hidden_size * num_directions if i > 0 else input_size
        # if batch_norm is True then batch norm layers should correspond to the
        # layers with an odd index
        if i % 2 == 0 or not batch_norm:
            # RNN layers corresponds to even indexes but also odd ones if
            # batch_norm is False
            if rnn_type == RNNType.LSTM:
                assert isinstance(rnn.rnn[i], torch.nn.LSTM)
            elif rnn_type == RNNType.GRU:
                assert isinstance(rnn.rnn[i], torch.nn.GRU)
            elif rnn_type == RNNType.BASIC_RNN:
                assert isinstance(rnn.rnn[i], torch.nn.RNN)
            else:
                raise ValueError(f"rnn_type {rnn_type} not supported by test")

            assert rnn.rnn[i].input_size == input_features
            assert rnn.rnn[i].hidden_size == hidden_size
            assert rnn.rnn[i].bias == bias
            assert not rnn.rnn[i].batch_first
            assert rnn.rnn[i].bidirectional == bidirectional

            if not (
                rnn_type == RNNType.LSTM
                and bias
                and forget_gate_bias is not None
            ):
                continue

            bias_value = getattr(rnn.rnn[i], f"bias_ih_l0")[
                hidden_size : 2 * hidden_size
            ]
            bias_value += getattr(rnn.rnn[i], f"bias_hh_l0")[
                hidden_size : 2 * hidden_size
            ]
            assert torch.allclose(bias_value, torch.tensor(forget_gate_bias))

        else:
            # i % 2 != 0 and batch_norm
            assert isinstance(rnn.rnn[i], torch.nn.BatchNorm1d)


@given(rnn_type=st.integers(min_value=100, max_value=300))
def test_error_raised_for_unknown_rnn_type(rnn_type: int) -> None:
    """Ensures error raised when unknown RNNType used."""
    assume(rnn_type not in list(RNNType))
    with pytest.raises(ValueError):
        RNN(rnn_type=rnn_type, input_size=1, hidden_size=1)  # type: ignore
