import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.rnn import build
from myrtlespeech.model.encoder_decoder.encoder.rnn import RNN
from myrtlespeech.protos import rnn_pb2
from tests.protos.test_rnn import rnns
from tests.utils.utils import tensors


# Utilities -------------------------------------------------------------------


def rnn_match_cfg(rnn: RNN, rnn_cfg: rnn_pb2.RNN, input_features: int) -> None:
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
    assert not rnn.rnn.batch_first
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
        bias += getattr(rnn.rnn, f"bias_hh_l{l}")[hidden_size : 2 * hidden_size]
        assert torch.allclose(bias, torch.tensor(forget_gate_bias))


# Tests -----------------------------------------------------------------------


@given(rnn_cfg=rnns(), input_features=st.integers(1, 128))
def test_build_rnn_returns_correct_rnn_with_valid_params(
    rnn_cfg: rnn_pb2.RNN, input_features: int
) -> None:
    """Test that build_rnn returns the correct RNN with valid params."""
    rnn = build(rnn_cfg, input_features)
    rnn_match_cfg(rnn, rnn_cfg, input_features)


@given(rnn_cfg=rnns(), tensor=tensors(min_n_dims=3, max_n_dims=3))
def test_build_rnn_rnn_forward_output_correct_size(
    rnn_cfg: rnn_pb2.RNN, tensor: torch.Tensor
) -> None:
    """Ensures returned RNN forward produces output with correct size."""
    seq_len, batch, input_features = tensor.size()
    rnn = build(rnn_cfg, input_features)

    out, _ = rnn(tensor)
    out_seq_len, out_batch, out_features = out.size()

    assert out_seq_len == seq_len
    assert out_batch == batch

    expected_out_features = rnn_cfg.hidden_size
    if rnn_cfg.bidirectional:
        expected_out_features *= 2
    assert out_features == expected_out_features


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
