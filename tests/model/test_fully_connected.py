from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from myrtlespeech.model.fully_connected import FullyConnected

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
    eps = 1e-8
    kwargs = {}
    kwargs["in_features"] = draw(st.integers(1, 32))
    kwargs["out_features"] = draw(st.integers(1, 32))
    kwargs["num_hidden_layers"] = draw(st.integers(0, 8))
    kwargs["dropout"] = None
    use_dropout = draw(st.booleans())
    if kwargs["num_hidden_layers"] == 0:
        kwargs["hidden_size"] = None
        kwargs["hidden_activation_fn"] = None
    else:
        kwargs["hidden_size"] = draw(st.integers(1, 32))
        kwargs["hidden_activation_fn"] = draw(
            st.sampled_from([torch.nn.ReLU(), torch.nn.Tanh()])
        )
        if use_dropout:
            kwargs["dropout"] = draw(st.floats(eps, 1 - eps, allow_nan=False))
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
    dropout = fully_connected.dropout
    fully_connected = fully_connected.fully_connected  # get torch module
    in_features = kwargs["in_features"]

    if kwargs["num_hidden_layers"] == 0:
        assert isinstance(fully_connected, torch.nn.Linear)
        assert fully_connected.in_features == kwargs["in_features"]
        assert fully_connected.out_features == kwargs["out_features"]
        assert dropout is None
        return

    assert isinstance(fully_connected, torch.nn.Sequential)

    # configuration of each layer in Sequential depends on whether
    # dropout is present (activation is always present)
    expected_len = 2 * kwargs["num_hidden_layers"] + 1

    if dropout is not None:
        expected_len += kwargs["num_hidden_layers"]

    assert len(fully_connected) == expected_len
    # Now check that the linear/activation_fn/dropout layers appear in the
    # expected order. We set the `module_idx` and then check for the
    # following condition:
    # if module_idx % total_types == <module_type>_idx:
    #     assert isinstance(module, <module_type>)
    if dropout is None:
        linear_idx = 0
        activation_idx = 1
        dropout_idx = -1  # infeasible value
        total_types = 2  # (activation and linear layer)
    else:
        linear_idx = 0
        activation_idx = 1
        dropout_idx = 2
        total_types = 3  # (activation, linear layer and dropout)

    for module_idx, module in enumerate(fully_connected):
        if module_idx % total_types == linear_idx:
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == in_features
            if module_idx == len(fully_connected) - 1:
                assert module.out_features == kwargs["out_features"]
            else:
                assert module.out_features == kwargs["hidden_size"]
            in_features = kwargs["hidden_size"]
        elif module_idx % total_types == activation_idx:
            assert isinstance(module, type(kwargs["hidden_activation_fn"]))
        elif module_idx % total_types == dropout_idx:
            assert isinstance(module, torch.nn.Dropout)
            assert abs(module.p - kwargs["dropout"]) < 1e-8
        else:
            raise NotImplementedError(
                "Issue with implementation. This branch \
                should not be hit!"
            )


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

    _, act_seq_lens = fully_connected((tensor, in_seq_lens))

    assert torch.all(act_seq_lens == in_seq_lens.to(act_seq_lens.device))


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
    dropout=st.floats(0.1, 1.0),
)
def test_fully_connected_raises_value_error_zero_hidden_dropout_not_None(
    fully_connected_kwargs: Tuple[FullyConnected, Dict], dropout: float
) -> None:
    """Ensures ValueError raised when hidden_size=0 and dropout is not None."""
    _, kwargs = fully_connected_kwargs
    kwargs["num_hidden_layers"] = 0
    kwargs["dropout"] = dropout
    with pytest.raises(ValueError):
        FullyConnected(**kwargs)


@given(
    fully_connected_kwargs=fully_connecteds(return_kwargs=True),
    dropout=st.floats(allow_nan=False, allow_infinity=False,),
)
def test_fully_connected_raises_value_error_dropout_negative_or_greater_than_1(
    fully_connected_kwargs: Tuple[FullyConnected, Dict], dropout: float
) -> None:
    """Ensures ValueError raised when dropout is < 0 or > 1."""
    _, kwargs = fully_connected_kwargs
    if dropout >= 0.0 and dropout <= 1.0:
        dropout += 1.1
    kwargs["dropout"] = dropout
    if kwargs["num_hidden_layers"] == 0:
        kwargs["num_hidden_layers"] = 1
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
    tensor=tensors(min_n_dims=3, max_n_dims=3),
)
def test_fully_connected_forward_returns_correct_size(
    fully_connected_kwargs: Tuple[FullyConnected, Dict], tensor: torch.Tensor
) -> None:
    # create new FullyConnected that accepts in_features sized input
    _, kwargs = fully_connected_kwargs
    kwargs["in_features"] = tensor.size()[-1]
    fully_connected = FullyConnected(**kwargs)

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
    assert out_size[-1] == kwargs["out_features"]
