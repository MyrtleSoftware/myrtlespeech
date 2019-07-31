from typing import Tuple, Union, Dict

import pytest
import torch
import hypothesis.strategies as st
from hypothesis import given

from myrtlespeech.model.encoder_decoder.decoder.fully_connected import (
    FullyConnected,
)
from tests.utils.utils import tensors


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def fully_connecteds(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[FullyConnected],
    st.SearchStrategy[Tuple[FullyConnected, Dict]],
]:
    """Returns a SearchStrategy for FullyConnected."""
    kwargs = {}
    kwargs["in_features"] = draw(st.integers(1, 32))
    kwargs["out_features"] = draw(st.integers(1, 32))
    kwargs["num_hidden_layers"] = draw(st.integers(0, 8))
    if kwargs["num_hidden_layers"] == 0:
        kwargs["hidden_size"] = None
        kwargs["hidden_activation_fn"] = None
    else:
        kwargs["hidden_size"] = draw(st.integers(1, 32))
        kwargs["hidden_activation_fn"] = draw(
            st.sampled_from([torch.nn.ReLU(), torch.nn.Tanh()])
        )
    if not return_kwargs:
        return FullyConnected(**kwargs)
    return FullyConnected(**kwargs), kwargs


# Tests -----------------------------------------------------------------------


@given(fully_connected_kwargs=fully_connecteds(return_kwargs=True))
def test_fully_connected_module_structure_correct_for_valid_kwargs(
    fully_connected_kwargs: Tuple[FullyConnected, Dict]
):
    """Ensures FullyConnected.fully_connected structure is correct."""
    fully_connected, kwargs = fully_connected_kwargs
    fully_connected = fully_connected.fully_connected  # get torch module

    if kwargs["num_hidden_layers"] == 0:
        assert isinstance(fully_connected, torch.nn.Linear)
        assert fully_connected.in_features == kwargs["in_features"]
        assert fully_connected.out_features == kwargs["out_features"]
        return

    assert isinstance(fully_connected, torch.nn.Sequential)
    assert len(fully_connected) == 2 * kwargs["num_hidden_layers"] + 1

    in_features = kwargs["in_features"]
    for idx, module in enumerate(fully_connected):
        # should be alternating linear/activation_fn layers
        if idx % 2 == 0:
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == in_features
            if idx == len(fully_connected) - 1:
                assert module.out_features == kwargs["out_features"]
            else:
                assert module.out_features == kwargs["hidden_size"]
            in_features = kwargs["hidden_size"]
        else:
            assert isinstance(module, type(kwargs["hidden_activation_fn"]))


@given(
    fully_connected=fully_connecteds(),
    batch_size=st.integers(min_value=1, max_value=16),
    max_seq_len=st.integers(min_value=1, max_value=64),
)
def test_fully_connected_module_returns_correct_seq_lens(
    fully_connected: FullyConnected, batch_size: int, max_seq_len: int
):
    """Ensures FullyConnected returns correct seq_lens when support enabled."""
    tensor = torch.empty(
        [max_seq_len, batch_size, fully_connected.in_features],
        requires_grad=False,
    ).normal_()

    in_seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )

    _, act_seq_lens = fully_connected(tensor, seq_lens=in_seq_lens)

    assert torch.all(act_seq_lens == in_seq_lens)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    num_hidden_layers=st.integers(-1000, -1),
)
def test_fully_connected_raises_value_error_negative_num_hidden_layers(
    fully_connected_kwargs: Tuple[FullyConnected, Dict], num_hidden_layers: int
) -> None:
    """Ensures ValueError raised when num_hidden_layers < 0."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = num_hidden_layers
    with pytest.raises(ValueError):
        FullyConnected(**kwargs)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    hidden_size=st.integers(1, 1000),
)
def test_fully_connected_raises_value_error_hidden_size_not_none(
    fully_connected_kwargs: Tuple[FullyConnected, Dict], hidden_size: int
) -> None:
    """Ensures ValueError raised when no hidden layers and not hidden_size."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = 0
    kwargs["hidden_size"] = hidden_size
    with pytest.raises(ValueError):
        FullyConnected(**kwargs)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    hidden_activation_fn=st.sampled_from([torch.nn.ReLU(), torch.nn.Tanh()]),
)
def test_fully_connected_raises_value_error_hidden_activation_fn_not_none(
    fully_connected_kwargs: Tuple[FullyConnected, Dict],
    hidden_activation_fn: torch.nn.Module,
) -> None:
    """Ensures ValueError raised when no hidden layers and no act fn."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = 0
    kwargs["hidden_activation_fn"] = hidden_activation_fn
    with pytest.raises(ValueError):
        FullyConnected(**kwargs)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    tensor=tensors(min_n_dims=2, max_n_dims=5),
)
def test_fully_connected_forward_returns_correct_size(
    fully_connected_kwargs: Tuple[FullyConnected, Dict], tensor: torch.Tensor
) -> None:
    # create new FullyConnected that accepts in_features sized input
    _, kwargs = fully_connected_kwargs
    kwargs["in_features"] = tensor.size()[-1]
    fully_connected = FullyConnected(**kwargs)
    out = fully_connected(tensor)

    in_size = tensor.size()
    out_size = out.size()

    assert len(in_size) == len(out_size)
    assert in_size[:-1] == out_size[:-1]
    assert out_size[-1] == kwargs["out_features"]
