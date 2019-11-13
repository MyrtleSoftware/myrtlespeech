import hypothesis.strategies as st
import torch
import pytest
from typing import Dict
from typing import Tuple
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.fully_connected import build
from myrtlespeech.model.fully_connected import FullyConnected
from myrtlespeech.protos import fully_connected_pb2

from tests.builders.test_activation import activation_match_cfg
from tests.protos.test_fully_connected import fully_connecteds
from tests.utils.utils import tensors


# Utilities -------------------------------------------------------------------


def fully_connected_module_match_cfg(
        fully_connected: FullyConnected,
        fully_connected_cfg: fully_connected_pb2.FullyConnected,
        input_features: int,
        output_features: int,
) -> None:
    """Ensures ``FullyConnected`` module matches protobuf configuration."""
    # otherwise it will be a Sequential of layers
    assert isinstance(fully_connected, torch.nn.Sequential)

    # configuration of each layer in Sequential depends on whether activation
    # and batch norm are present
    act_fn_is_none = fully_connected_cfg.activation.HasField("identity")
    batch_norm = fully_connected_cfg.batch_norm
    hidden_size = fully_connected_cfg.hidden_size

    assert len(fully_connected) == fully_connected_cfg.num_hidden_layers + 1

    for idx, module in enumerate(fully_connected):
        # should be alternating linear/activation_fn layers if !act_fn_is_none
        # or linear/batch_norm_activation_fn if also batch_norm is True
        assert isinstance(module.fully_connected, torch.nn.Linear)
        assert module.fully_connected.in_features == input_features
        if idx == len(fully_connected) - 1:
            assert module.fully_connected.out_features == output_features
        else:
            assert module.fully_connected.out_features == hidden_size
        input_features = hidden_size

        if batch_norm and idx < fully_connected_cfg.num_hidden_layers:
            assert isinstance(module.batch_norm, torch.nn.BatchNorm1d)
        if not act_fn_is_none and idx < fully_connected_cfg.num_hidden_layers:
            activation_match_cfg(module.activation,
                                 fully_connected_cfg.activation)


# Tests -----------------------------------------------------------------------


@given(
    fully_connected_cfg=fully_connecteds(),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
)
def test_build_fully_connected_returns_correct_module_structure(
        fully_connected_cfg: fully_connected_pb2.FullyConnected,
        input_features: int,
        output_features: int,
) -> None:
    """Ensures Module returned has correct structure."""
    if fully_connected_cfg.num_hidden_layers == 0:
        assume(fully_connected_cfg.hidden_size is None)
        assume(fully_connected_cfg.batch_norm is None)
        assume(fully_connected_cfg.activation is None)

    actual = build(fully_connected_cfg, input_features, output_features)
    fully_connected_module_match_cfg(
        actual, fully_connected_cfg, input_features, output_features
    )


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
    num_hidden_layers=st.integers(-1000, -1),
)
def test_fully_connected_raises_value_error_negative_num_hidden_layers(
        fully_connected_kwargs: Tuple[FullyConnected, Dict],
        input_features: int,
        output_features: int,
        num_hidden_layers: int
) -> None:
    """Ensures ValueError raised when num_hidden_layers < 0."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = num_hidden_layers
    with pytest.raises(ValueError):
        build(fully_connected_pb2.FullyConnected(**kwargs),
              input_features, output_features)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
    hidden_size=st.integers(1, 1000),
)
def test_fully_connected_raises_value_error_hidden_size_not_none(
        fully_connected_kwargs: Tuple[FullyConnected, Dict],
        input_features: int,
        output_features: int,
        hidden_size: int
) -> None:
    """Ensures ValueError raised when no hidden layers and not hidden_size."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = 0
    kwargs["hidden_size"] = hidden_size
    with pytest.raises(ValueError):
        build(fully_connected_pb2.FullyConnected(**kwargs),
              input_features, output_features)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
    hidden_activation_fn=st.sampled_from([torch.nn.ReLU(),
                                          torch.nn.Hardtanh(min_val=0.0,
                                                            max_val=20.0)]),
)
def test_fully_connected_raises_value_error_hidden_activation_fn_not_none(
        fully_connected_kwargs: Tuple[FullyConnected, Dict],
        input_features: int,
        output_features: int,
        hidden_activation_fn: torch.nn.Module,
) -> None:
    """Ensures ValueError raised when no hidden layers and no act fn."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = 0
    kwargs["hidden_activation_fn"] = hidden_activation_fn
    with pytest.raises(ValueError):
        build(fully_connected_pb2.FullyConnected(**kwargs),
              input_features, output_features)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True,
                                            valid_only=True),
    output_features=st.integers(min_value=1, max_value=32),
    tensor=tensors(min_n_dims=3, max_n_dims=3),
)
def test_fully_connected_forward_returns_correct_size(
        fully_connected_kwargs: Tuple[FullyConnected, Dict],
        output_features: int,
        tensor: torch.Tensor
) -> None:
    # create new FullyConnected that accepts in_features sized input
    _, kwargs = fully_connected_kwargs
    # if batch_size == 1 don't add batch norm to the fully connected network
    if tensor.size(1) == 1:
        kwargs["batch_norm"] = False

    fully_connected = build(fully_connected_pb2.FullyConnected(**kwargs),
                            tensor.size()[-1], output_features)

    max_seq_len, batch_size, *_ = tensor.size()
    in_seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )
    out, _ = fully_connected((tensor, in_seq_lens))

    in_size = tensor.size()
    out_size = out.size()

    assert len(in_size) == len(out_size)
    assert in_size[:-1] == out_size[:-1]
    assert out_size[-1] == output_features


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True,
                                            valid_only=True),
    in_features=st.integers(min_value=1, max_value=32),
    out_features=st.integers(min_value=1, max_value=32),
    batch_size=st.integers(min_value=1, max_value=16),
    max_seq_len=st.integers(min_value=1, max_value=64),
)
def test_fully_connected_module_returns_correct_seq_lens(
        fully_connected_kwargs: Tuple[FullyConnected, Dict],
        in_features: int, out_features: int,
        batch_size: int, max_seq_len: int
):
    """Ensures FullyConnected returns correct seq_lens when support enabled."""
    # create new FullyConnected that accepts in_features sized input
    _, kwargs = fully_connected_kwargs
    # if batch_size == 1 don't add batch norm to the fully connected network
    kwargs["batch_norm"] = batch_size > 1

    fully_connected = build(fully_connected_pb2.FullyConnected(**kwargs),
                            in_features, out_features)

    tensor = torch.empty(
        [max_seq_len, batch_size, fully_connected[0].in_features],
        requires_grad=False,
    ).normal_()

    in_seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )

    _, act_seq_lens = fully_connected((tensor, in_seq_lens))

    assert torch.all(act_seq_lens == in_seq_lens)
