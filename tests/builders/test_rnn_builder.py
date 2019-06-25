import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.rnn_builder import build_rnn
from myrtlespeech.protos import rnn_pb2
from tests.protos.test_rnn import rnns


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_input_sizes(draw) -> st.SearchStrategy[torch.Size]:
    """Returns a SearchStrategy over valid RNN input_sizes."""
    batch_size = draw(st.one_of(st.just(-1), st.integers(1, 128)))
    seq_len = draw(st.one_of(st.just(-1), st.integers(1, 1000)))
    n_features = draw(st.integers(1, 1024))
    return torch.Size([batch_size, seq_len, n_features])


# Tests -----------------------------------------------------------------------


@given(rnn_cfg=rnns(), input_size=rnn_input_sizes())
def test_build_rnn_returns_correct_rnn_with_valid_params(
    rnn_cfg: rnn_pb2.RNN, input_size: torch.Size
) -> None:
    """Test that build_rnn returns the correct RNN with valid params."""
    rnn, _ = build_rnn(rnn_cfg, input_size)

    if rnn_cfg.rnn_type == rnn_pb2.RNN.RNN_TYPE.LSTM:
        assert isinstance(rnn, torch.nn.LSTM)
    elif rnn_cfg.rnn_type == rnn_pb2.RNN.RNN_TYPE.GRU:
        assert isinstance(rnn, torch.nn.GRU)
    else:
        assert isinstance(rnn, torch.nn.RNN)

    assert rnn.input_size == input_size[2]
    assert rnn.hidden_size == rnn_cfg.hidden_size
    assert rnn.num_layers == rnn_cfg.num_layers
    assert rnn.bias == rnn_cfg.bias
    assert rnn.batch_first
    assert rnn.bidirectional == rnn_cfg.bidirectional


@given(rnn_cfg=rnns(), input_size=rnn_input_sizes())
def test_build_rnn_returns_correct_output_size(
    rnn_cfg: rnn_pb2.RNN, input_size: torch.Size
) -> None:
    """Test that build_rnn returns the correct output size."""
    _, output_size = build_rnn(rnn_cfg, input_size)
    assert output_size[0] == input_size[0]
    assert output_size[1] == input_size[1]

    out_features = rnn_cfg.hidden_size
    if rnn_cfg.bidirectional:
        out_features *= 2
    assert output_size[2] == out_features


@given(rnn_cfg=rnns())
def test_non_3_dimensional_input_size_raises_value_error(
    rnn_cfg: rnn_pb2.RNN
) -> None:
    """Ensures ValueError is raised when input_size is not 3-dimensional."""
    with pytest.raises(ValueError):
        build_rnn(rnn_cfg, input_size=torch.Size([]))


@given(
    rnn_cfg=rnns(),
    input_size=rnn_input_sizes(),
    invalid_rnn_type=st.integers(0, 128),
)
def test_unknown_rnn_type_raises_value_error(
    rnn_cfg: rnn_pb2.RNN, input_size: torch.Size, invalid_rnn_type: int
) -> None:
    """Ensures ValueError is raised when rnn_type not supported.

    This can occur when the protobuf is updated and build_rnn is not.
    """
    assume(invalid_rnn_type not in rnn_pb2.RNN.RNN_TYPE.values())
    rnn_cfg.rnn_type = invalid_rnn_type
    with pytest.raises(ValueError):
        build_rnn(rnn_cfg, input_size=input_size)
