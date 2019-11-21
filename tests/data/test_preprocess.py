from typing import Dict

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from myrtlespeech.data.preprocess import AddSequenceLength
from myrtlespeech.data.preprocess import SpecAugment

from tests.utils.utils import tensors


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def spec_augments(draw) -> st.SearchStrategy[SpecAugment]:
    """Returns a SearchStrategy for SpecAugment."""
    kwargs: Dict = {}
    kwargs["feature_mask"] = draw(st.integers(0, 30))
    kwargs["time_mask"] = draw(st.integers(0, 30))
    kwargs["n_feature_masks"] = draw(st.integers(0, 3))
    kwargs["n_time_masks"] = draw(st.integers(0, 3))
    spec_augment = SpecAugment(**kwargs)
    return spec_augment


# Tests -----------------------------------------------------------------------


@given(data=st.data(), tensor=tensors(min_n_dims=1))
def test_add_sequence_length_returns_correct_seq_len(
    data, tensor: torch.Tensor
) -> None:
    """Ensures AddSequenceLength returns correct sequence length."""
    length_dim = data.draw(
        st.integers(min_value=0, max_value=len(tensor.size()) - 1)
    )

    add_seq_len = AddSequenceLength(length_dim=length_dim)

    out, seq_len = add_seq_len(tensor)

    assert torch.all(out == tensor)
    assert seq_len == torch.tensor([tensor.size(length_dim)])


# SpecAugment ---------------------------


@given(
    spec_augment=spec_augments(),
    tensor=tensors(
        min_n_dims=3, max_n_dims=3, min_dim_size=1, max_dim_size=100
    ),
)
def test_spec_augment_returns_tensor_same_shape(
    spec_augment: SpecAugment, tensor: torch.Tensor
) -> None:
    """Ensures SpecAugment returns a tensor with the same shape."""
    out = spec_augment(tensor)
    assert out.size() == tensor.size()


@given(
    sa=spec_augments(),
    tensor=tensors(
        min_n_dims=3, max_n_dims=3, min_dim_size=1, max_dim_size=100
    ),
)
def test_spec_augment_n_zeros_less_than_max(
    sa: SpecAugment, tensor: torch.Tensor
) -> None:
    """Ensures number of parameters zeroed by SpecAugment is less than the max.

    The maximum number is:

        channels*(
            n_feature_masks*feature_mask*time_steps +
            n_time_masks*time_mask*features
        )
    """
    tensor.fill_(1)  # ensure no zeros before SpecAugment applied
    channels, features, time_steps = tensor.size()
    out = sa(tensor)
    max = sa.n_feature_masks * sa.feature_mask * time_steps
    max += sa.n_time_masks * sa.time_mask * features
    max *= channels
    assert (out == 0).sum() <= max


def test_spec_augment_raises_value_error_invalid_params() -> None:
    """Ensures ValueError raised when parameters less than zero."""
    with pytest.raises(ValueError):
        SpecAugment(feature_mask=-1, time_mask=1)

    with pytest.raises(ValueError):
        SpecAugment(feature_mask=1, time_mask=-1)

    with pytest.raises(ValueError):
        SpecAugment(feature_mask=1, time_mask=1, n_feature_masks=-1)

    with pytest.raises(ValueError):
        SpecAugment(feature_mask=1, time_mask=1, n_time_masks=-1)
