import hypothesis.strategies as st
import pytest
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.transducer_loss import build
from myrtlespeech.loss.transducer_loss import TransducerLoss
from myrtlespeech.protos import transducer_loss_pb2

from tests.protos.test_transducer_loss import transducer_losses


# Utilities -------------------------------------------------------------------


def transducer_loss_module_match_cfg(
    transducer_loss: TransducerLoss,
    transducer_loss_cfg: transducer_loss_pb2.TransducerLoss,
) -> None:
    """Ensures TransducerLoss matches protobuf configuration."""
    assert isinstance(transducer_loss, TransducerLoss)
    assert hasattr(transducer_loss, "transducer_loss")
    assert hasattr(transducer_loss, "use_cuda")

    # verify torch module attributes
    transducer_loss = transducer_loss.transducer_loss
    assert transducer_loss.blank == transducer_loss_cfg.blank_index
    print(transducer_loss_cfg)
    if (
        transducer_loss_cfg.reduction
        == transducer_loss_pb2.TransducerLoss.NONE
    ):
        assert transducer_loss.reduction == "none"
    elif (
        transducer_loss_cfg.reduction
        == transducer_loss_pb2.TransducerLoss.MEAN
    ):
        assert transducer_loss.reduction == "mean"
    elif (
        transducer_loss_cfg.reduction == transducer_loss_pb2.TransducerLoss.SUM
    ):
        assert transducer_loss.reduction == "sum"
    else:
        raise ValueError(f"unknown reduction {transducer_loss.reduction}")


# Tests -----------------------------------------------------------------------


@given(transducer_loss_cfg=transducer_losses())
def test_build_returns_correct_transducer_loss_with_valid_params(
    transducer_loss_cfg: transducer_loss_pb2.TransducerLoss,
) -> None:
    """Test that build returns the correct TransducerLoss with valid params."""
    transducer_loss = build(transducer_loss_cfg)
    transducer_loss_module_match_cfg(transducer_loss, transducer_loss_cfg)


@given(
    transducer_loss_cfg=transducer_losses(),
    invalid_reduction=st.integers(0, 128),
)
def test_unknown_reduction_raises_value_error(
    transducer_loss_cfg: transducer_loss_pb2.TransducerLoss,
    invalid_reduction: int,
) -> None:
    """Ensures ValueError is raised when reduction not supported.

    This can occur when the protobuf is updated and build is not.
    """
    assume(
        invalid_reduction
        not in transducer_loss_pb2.TransducerLoss.REDUCTION.values()
    )
    transducer_loss_cfg.reduction = invalid_reduction  # type: ignore
    with pytest.raises(ValueError):
        build(transducer_loss_cfg)
