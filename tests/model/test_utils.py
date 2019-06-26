from typing import Tuple, Callable

import torch
import pytest
from hypothesis import given

from myrtlespeech.model.utils import Lambda

from tests.utils.utils import tensors


# Fixtures and Strategies -----------------------------------------------------


@pytest.fixture(params=[lambda x: x * 2, lambda x: x + 1])
def lambda_fn(request) -> Tuple[str, Callable]:
    return request.param


# Tests -----------------------------------------------------------------------


@given(tensor=tensors())
def test_lambda_module_applies_lambda_fn(
    lambda_fn: Callable, tensor: torch.Tensor
) -> None:
    """Ensures Lambda Module applies given lambda_fn to input."""
    lambda_module = Lambda(lambda_fn)
    assert torch.all(lambda_module(tensor) == lambda_fn(tensor))
