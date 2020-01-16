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
        assert not fully_connected.HasField("dropout")
        return

    # otherwise it will be a Sequential of layers
    assert isinstance(fully_connected, torch.nn.Sequential)

    # expected configuration of each layer in Sequential depends on whether
    # both/either of {activation, dropout} are present.
    act_fn_is_none = fully_connected_cfg.activation.HasField("identity")
    dropout_is_none = not fully_connected_cfg.HasField("dropout")
    if act_fn_is_none:
        expected_len = fully_connected_cfg.num_hidden_layers + 1
    else:
        expected_len = 2 * fully_connected_cfg.num_hidden_layers + 1

    if not dropout_is_none:
        expected_len += fully_connected_cfg.num_hidden_layers

    assert len(fully_connected) == expected_len

    # Now check that the linear/activation_fn/dropout layers appear in the
    # expected order. We set the ``module_idx`` and then check for the
    # following condition:
    # if module_idx % total_types == <module_type>_idx:
    #     assert isinstance(module, <module_type>)
    linear_idx = 0  # in all cases
    activation_idx = -1  # infeasible value as default
    dropout_idx = -1
    if act_fn_is_none and dropout_is_none:
        total_types = 1  # (linear layers only)
    elif not act_fn_is_none and dropout_is_none:
        total_types = 2  # (linear and activation)
        activation_idx = 1
    elif act_fn_is_none and not dropout_is_none:
        total_types = 2
        dropout_idx = 1
    elif not act_fn_is_none and not dropout_is_none:
        total_types = 3
        activation_idx = 1
        dropout_idx = 2

    for module_idx, module in enumerate(fully_connected):
        if module_idx % total_types == linear_idx:
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == input_features
            if module_idx == len(fully_connected) - 1:
                assert module.out_features == output_features
            else:
                assert module.out_features == fully_connected_cfg.hidden_size
            input_features = fully_connected_cfg.hidden_size
        elif module_idx % total_types == activation_idx:
            activation_match_cfg(module, fully_connected_cfg.activation)
        elif module_idx % total_types == dropout_idx:
            assert isinstance(module, torch.nn.Dropout)
            assert abs(module.p - fully_connected_cfg.dropout.value) < 1e-8
        else:
            raise ValueError(
                "Check module_idx and total_types assignment. It "
                "**should not** be possible to hit this branch!"
            )


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
