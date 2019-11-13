from typing import Tuple

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.rnn import build
from myrtlespeech.model.rnn import RNN
from myrtlespeech.protos import rnn_pb2

from tests.protos.test_rnn import rnns
from tests.utils.utils import tensors


# Utilities -------------------------------------------------------------------


def rnn_match_cfg(rnn: RNN, rnn_cfg: rnn_pb2.RNN, input_features: int,
                  layer_idx: int) -> None:
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
    assert rnn_cfg.bias == rnn.rnn.bias
    assert not rnn.rnn.batch_first
    assert rnn.rnn.dropout == 0.0
    assert rnn_cfg.bidirectional == rnn.rnn.bidirectional

    if rnn_cfg.batch_norm and layer_idx > 0:
        assert isinstance(rnn.batch_norm, torch.nn.BatchNorm1d)

    if not (
            rnn_cfg.rnn_type == rnn_pb2.RNN.LSTM
            and rnn_cfg.bias
            and rnn_cfg.HasField("forget_gate_bias")
    ):
        return

    hidden_size = rnn_cfg.hidden_size
    forget_gate_bias = rnn_cfg.forget_gate_bias.value
    bias = getattr(rnn.rnn, f"bias_ih_l0")[hidden_size: 2 * hidden_size]
    bias += getattr(rnn.rnn, f"bias_hh_l0")[hidden_size: 2 * hidden_size]
    assert torch.allclose(bias, torch.tensor(forget_gate_bias))


@st.composite
def rnn_cfg_tensors(
        draw
) -> st.SearchStrategy[Tuple[torch.nn.Module, rnn_pb2.RNN, torch.Tensor]]:
    """Returns a search strategy for RNNs built from a config + valid input."""
    rnn_cfg = draw(rnns())
    tensor = draw(tensors(min_n_dims=3, max_n_dims=3))
    if tensor.size(1) == 1:
        # if batch_size == 1 don't add batch norm to the rnn
        rnn_cfg.batch_norm = False
    rnn, _ = build(rnn_cfg, input_features=tensor.size(2))
    return rnn, rnn_cfg, tensor


# Tests -----------------------------------------------------------------------


@given(rnn_cfg=rnns(), input_features=st.integers(1, 128))
def test_build_rnn_returns_correct_rnn_with_valid_params(
        rnn_cfg: rnn_pb2.RNN, input_features: int
) -> None:
    """Test that build_rnn returns the correct RNN with valid params."""
    rnn, rnn_output_size = build(rnn_cfg, input_features)

    for i in range(rnn_cfg.num_layers):
        num_directions = 2 if rnn_cfg.bidirectional else 1
        rnn_match_cfg(rnn[i], rnn_cfg, rnn_cfg.hidden_size * num_directions
                      if i > 0 else input_features, i)


@given(rnn_cfg_tensor=rnn_cfg_tensors())
def test_build_rnn_rnn_forward_output_correct_size(
        rnn_cfg_tensor: Tuple[torch.nn.Module, rnn_pb2.RNN, torch.Tensor]
) -> None:
    """Ensures returned RNN forward produces output with correct size."""
    rnn, rnn_cfg, tensor = rnn_cfg_tensor
    seq_len, batch, input_features = tensor.size()

    in_seq_lens = torch.randint(low=1, high=1 + seq_len, size=(batch,))

    out, out_seq_lens = rnn((tensor, in_seq_lens))
    out_seq_len, out_batch, out_features = out.size()

    assert out_seq_len == seq_len
    assert out_batch == batch
    expected_out_features = rnn_cfg.hidden_size
    if rnn_cfg.bidirectional:
        expected_out_features *= 2
    assert out_features == expected_out_features

    assert torch.all(in_seq_lens == out_seq_lens)


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
