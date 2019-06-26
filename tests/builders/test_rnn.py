import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.rnn import build_rnn
from myrtlespeech.protos import rnn_pb2
from tests.protos.test_rnn import rnns
from tests.utils.utils import tensors


# Tests -----------------------------------------------------------------------


@given(rnn_cfg=rnns(), input_features=st.integers(1, 128))
def test_build_rnn_returns_correct_rnn_with_valid_params(
    rnn_cfg: rnn_pb2.RNN, input_features: int
) -> None:
    """Test that build_rnn returns the correct RNN with valid params."""
    rnn = build_rnn(rnn_cfg, input_features)

    if rnn_cfg.rnn_type == rnn_pb2.RNN.LSTM:
        assert isinstance(rnn, torch.nn.LSTM)
    elif rnn_cfg.rnn_type == rnn_pb2.RNN.GRU:
        assert isinstance(rnn, torch.nn.GRU)
    else:
        assert isinstance(rnn, torch.nn.RNN)

    assert rnn.input_size == input_features
    assert rnn.hidden_size == rnn_cfg.hidden_size
    assert rnn.num_layers == rnn_cfg.num_layers
    assert rnn.bias == rnn_cfg.bias
    assert not rnn.batch_first
    assert rnn.bidirectional == rnn_cfg.bidirectional


@given(rnn_cfg=rnns(), tensor=tensors(min_n_dims=3, max_n_dims=3))
def test_build_rnn_rnn_forward_output_correct_size(
    rnn_cfg: rnn_pb2.RNN, tensor: torch.Tensor
) -> None:
    """Ensures returned RNN forward produces output with correct size."""
    seq_len, batch, input_features = tensor.size()
    rnn = build_rnn(rnn_cfg, input_features)

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
        build_rnn(rnn_cfg, input_features=input_features)
