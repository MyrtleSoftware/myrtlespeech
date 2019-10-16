import torch
from hypothesis import given
from myrtlespeech.builders.activation import build
from myrtlespeech.protos import activation_pb2

from tests.protos.test_activation import activations


# Utilities -------------------------------------------------------------------


def activation_match_cfg(
    activation: torch.nn.Module, activation_cfg: activation_pb2.Activation
) -> None:
    """Ensures activation matches protobuf configuration."""
    if activation_cfg.HasField("identity"):
        assert isinstance(activation, torch.nn.Identity)
    elif activation_cfg.HasField("hardtanh"):
        assert isinstance(activation, torch.nn.Hardtanh)
        assert activation_cfg.hardtanh.min_val == activation.min_val
        assert activation_cfg.hardtanh.max_val == activation.max_val
    elif activation_cfg.HasField("relu"):
        assert isinstance(activation, torch.nn.ReLU)
    else:
        raise ValueError(f"unknown activation_cfg {activation_cfg}")


# Tests -----------------------------------------------------------------------


@given(activation_cfg=activations())
def test_build_activation_returns_correct_module(
    activation_cfg: activation_pb2.Activation,
) -> None:
    """Ensures returned activation has correct structure."""
    activation = build(activation_cfg)
    activation_match_cfg(activation, activation_cfg)
