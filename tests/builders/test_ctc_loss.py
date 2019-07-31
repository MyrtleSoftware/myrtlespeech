import hypothesis.strategies as st
import pytest
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.ctc_loss import build
from myrtlespeech.loss.ctc_loss import CTCLoss
from myrtlespeech.protos import ctc_loss_pb2

from tests.protos.test_ctc_loss import ctc_losses


# Utilities -------------------------------------------------------------------


def ctc_loss_module_match_cfg(
    ctc_loss: CTCLoss, ctc_loss_cfg: ctc_loss_pb2.CTCLoss
) -> None:
    """Ensures CTCLoss matches protobuf configuration."""
    assert isinstance(ctc_loss, CTCLoss)
    assert hasattr(ctc_loss, "ctc_loss")
    assert hasattr(ctc_loss, "log_softmax")

    # verify torch module attributes
    ctc_loss = ctc_loss.ctc_loss
    assert ctc_loss.blank == ctc_loss_cfg.blank_index

    if ctc_loss_cfg.reduction == ctc_loss_pb2.CTCLoss.NONE:
        assert ctc_loss.reduction == "none"
    elif ctc_loss_cfg.reduction == ctc_loss_pb2.CTCLoss.MEAN:
        assert ctc_loss.reduction == "mean"
    elif ctc_loss_cfg.reduction == ctc_loss_pb2.CTCLoss.SUM:
        assert ctc_loss.reduction == "sum"
    else:
        raise ValueError(f"unknown reduction {ctc_loss.reduction}")


# Tests -----------------------------------------------------------------------


@given(ctc_loss_cfg=ctc_losses())
def test_build_returns_correct_ctc_loss_with_valid_params(
    ctc_loss_cfg: ctc_loss_pb2.CTCLoss
) -> None:
    """Test that build returns the correct CTCLoss with valid params."""
    ctc_loss = build(ctc_loss_cfg)
    ctc_loss_module_match_cfg(ctc_loss, ctc_loss_cfg)


@given(ctc_loss_cfg=ctc_losses(), invalid_reduction=st.integers(0, 128))
def test_unknown_reduction_raises_value_error(
    ctc_loss_cfg: ctc_loss_pb2.CTCLoss, invalid_reduction: int
) -> None:
    """Ensures ValueError is raised when reduction not supported.

    This can occur when the protobuf is updated and build is not.
    """
    assume(invalid_reduction not in ctc_loss_pb2.CTCLoss.REDUCTION.values())
    ctc_loss_cfg.reduction = invalid_reduction  # type: ignore
    with pytest.raises(ValueError):
        build(ctc_loss_cfg)
