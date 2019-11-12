import hypothesis.strategies as st
import torch
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.fully_connected import build
from myrtlespeech.model.fully_connected import FullyConnected
from myrtlespeech.protos import fully_connected_pb2

from tests.builders.test_activation import activation_match_cfg
from tests.protos.test_fully_connected import fully_connecteds


# Utilities -------------------------------------------------------------------


def fully_connected_module_match_cfg(
    fully_connected: FullyConnected,
    fully_connected_cfg: fully_connected_pb2.FullyConnected,
    input_features: int,
    output_features: int,
) -> None:
    """Ensures ``FullyConnected`` module matches protobuf configuration."""
    fully_connected = fully_connected.fully_connected  # get torch module

    # if no hidden layers then test that the module is Linear with corret
    # sizes, ignore activation
    if fully_connected_cfg.num_hidden_layers == 0:
        assert isinstance(fully_connected, torch.nn.Linear)
        assert fully_connected.in_features == input_features
        assert fully_connected.out_features == output_features
        return

    # otherwise it will be a Sequential of layers
    assert isinstance(fully_connected, torch.nn.Sequential)

    # configuration of each layer in Sequential depends on whether activation
    # is present
    act_fn_is_none = fully_connected_cfg.activation.HasField("identity")
    dropout_is_used = fully_connected_cfg.dropout > 1e-8
    if act_fn_is_none:
        expected_len = fully_connected_cfg.num_hidden_layers + 1
    else:
        expected_len = 2 * fully_connected_cfg.num_hidden_layers + 1

    if dropout_is_used:
        expected_len += fully_connected_cfg.num_hidden_layers

    assert len(fully_connected) == expected_len

    for idx, module in enumerate(fully_connected):
        # should be linear/activation_fn layers if !act_fn_is_none
        if act_fn_is_none or ((not act_fn_is_none) and idx % 2 == 0):
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == input_features
            if idx == len(fully_connected) - 1:
                assert module.out_features == output_features
            else:
                assert module.out_features == fully_connected_cfg.hidden_size
            input_features = fully_connected_cfg.hidden_size
        elif not act_fn_is_none:
            activation_match_cfg(module, fully_connected_cfg.activation)


# Tests -----------------------------------------------------------------------


@given(
    fully_connected_cfg=fully_connecteds(),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
)
@settings(deadline=3000)
def test_build_fully_connected_returns_correct_module_structure(
    fully_connected_cfg: fully_connected_pb2.FullyConnected,
    input_features: int,
    output_features: int,
) -> None:
    """Ensures Module returned has correct structure."""
    if fully_connected_cfg.num_hidden_layers == 0:
        assume(fully_connected_cfg.hidden_size is None)
        assume(fully_connected_cfg.activation is None)

    actual = build(fully_connected_cfg, input_features, output_features)
    fully_connected_module_match_cfg(
        actual, fully_connected_cfg, input_features, output_features
    )
