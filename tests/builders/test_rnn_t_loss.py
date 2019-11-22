import hypothesis.strategies as st
import pytest
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.rnn_t_loss import build
from myrtlespeech.loss.rnn_t_loss import RNNTLoss
from myrtlespeech.protos import rnn_t_loss_pb2

from tests.protos.test_rnn_t_loss import rnn_t_losses


# Utilities -------------------------------------------------------------------


def rnn_t_loss_module_match_cfg(
    rnn_t_loss: RNNTLoss, rnn_t_loss_cfg: rnn_t_loss_pb2.RNNTLoss
) -> None:
    """Ensures RNNTLoss matches protobuf configuration."""
    assert isinstance(rnn_t_loss, RNNTLoss)
    assert hasattr(rnn_t_loss, "rnnt_loss")
    assert hasattr(rnn_t_loss, "use_cuda")

    # verify torch module attributes
    rnn_t_loss = rnn_t_loss.rnnt_loss
    assert rnn_t_loss.blank == rnn_t_loss_cfg.blank_index
    print(rnn_t_loss_cfg)
    if rnn_t_loss_cfg.reduction == rnn_t_loss_pb2.RNNTLoss.NONE:
        assert rnn_t_loss.reduction == "none"
    elif rnn_t_loss_cfg.reduction == rnn_t_loss_pb2.RNNTLoss.MEAN:
        assert rnn_t_loss.reduction == "mean"
    elif rnn_t_loss_cfg.reduction == rnn_t_loss_pb2.RNNTLoss.SUM:
        assert rnn_t_loss.reduction == "sum"
    else:
        raise ValueError(f"unknown reduction {rnn_t_loss.reduction}")


# Tests -----------------------------------------------------------------------


@given(rnn_t_loss_cfg=rnn_t_losses())
def test_build_returns_correct_rnn_t_loss_with_valid_params(
    rnn_t_loss_cfg: rnn_t_loss_pb2.RNNTLoss,
) -> None:
    """Test that build returns the correct RNNTLoss with valid params."""
    rnn_t_loss = build(rnn_t_loss_cfg)
    rnn_t_loss_module_match_cfg(rnn_t_loss, rnn_t_loss_cfg)


@given(rnn_t_loss_cfg=rnn_t_losses(), invalid_reduction=st.integers(0, 128))
def test_unknown_reduction_raises_value_error(
    rnn_t_loss_cfg: rnn_t_loss_pb2.RNNTLoss, invalid_reduction: int
) -> None:
    """Ensures ValueError is raised when reduction not supported.

    This can occur when the protobuf is updated and build is not.
    """
    assume(invalid_reduction not in rnn_t_loss_pb2.RNNTLoss.REDUCTION.values())
    rnn_t_loss_cfg.reduction = invalid_reduction  # type: ignore
    with pytest.raises(ValueError):
        build(rnn_t_loss_cfg)
