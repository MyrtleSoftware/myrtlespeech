import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.rnn import build
from myrtlespeech.model.rnn import RNN
from myrtlespeech.protos import rnn_pb2

from tests.protos.test_rnn import rnns


# Utilities -------------------------------------------------------------------


def rnn_match_cfg(
    rnn: RNN, rnn_cfg: rnn_pb2.RNN, input_features: int, batch_first: bool
) -> None:
    """Ensures RNN matches protobuf configuration."""
    if rnn_cfg.rnn_type == rnn_pb2.RNN.LSTM:
        assert isinstance(rnn.rnn, torch.nn.LSTM)
    elif rnn_cfg.rnn_type == rnn_pb2.RNN.GRU:
        assert isinstance(rnn.rnn, torch.nn.GRU)
    elif rnn_cfg.rnn_type == rnn_pb2.RNN.BASIC_RNN:
        assert isinstance(rnn.rnn, torch.nn.RNN)
    else:
        raise ValueError(f"rnn_type {rnn_cfg.rnn_type} not supported by test")

    assert input_features == rnn.rnn.input_size
    assert rnn_cfg.hidden_size == rnn.rnn.hidden_size
    assert rnn_cfg.num_layers == rnn.rnn.num_layers
    assert rnn_cfg.bias == rnn.rnn.bias
    assert batch_first == rnn.rnn.batch_first
    assert rnn.rnn.dropout == 0.0
    assert rnn_cfg.bidirectional == rnn.rnn.bidirectional

    if not (
        rnn_cfg.rnn_type == rnn_pb2.RNN.LSTM
        and rnn_cfg.bias
        and rnn_cfg.HasField("forget_gate_bias")
    ):
        return

    hidden_size = rnn_cfg.hidden_size
    forget_gate_bias = rnn_cfg.forget_gate_bias.value
    for l in range(rnn_cfg.num_layers):
        bias = getattr(rnn.rnn, f"bias_ih_l{l}")[hidden_size : 2 * hidden_size]
        bias += getattr(rnn.rnn, f"bias_hh_l{l}")[
            hidden_size : 2 * hidden_size
        ]
        assert torch.allclose(
            bias, torch.tensor(forget_gate_bias).to(bias.device)
        )


# Tests -----------------------------------------------------------------------


@given(
    rnn_cfg=rnns(),
    input_features=st.integers(1, 32),
    batch_first=st.booleans(),
)
def test_build_rnn_returns_correct_rnn_with_valid_params(
    rnn_cfg: rnn_pb2.RNN, input_features: int, batch_first: bool
) -> None:
    """Test that build_rnn returns the correct RNN with valid params."""
    rnn, rnn_output_size = build(rnn_cfg, input_features, batch_first)
    rnn_match_cfg(rnn, rnn_cfg, input_features, batch_first)


@given(
    rnn_cfg=rnns(),
    input_features=st.integers(1, 128),
    invalid_rnn_type=st.integers(0, 128),
)
def test_unknown_rnn_type_raises_value_error(
    rnn_cfg: rnn_pb2.RNN, input_features: int, invalid_rnn_type: int
) -> None:
    """Ensures ValueError is raised when rnn_type not supported.

    This can occur when the protobuf is updated and build_rnn is not.
    """
    assume(invalid_rnn_type not in rnn_pb2.RNN.RNN_TYPE.values())
    rnn_cfg.rnn_type = invalid_rnn_type  # type: ignore
    with pytest.raises(ValueError):
        build(rnn_cfg, input_features=input_features)
