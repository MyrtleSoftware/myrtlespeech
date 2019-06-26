"""Common utilities used in tests.

TODO: Mention that some of these functions are based on PyTorch's testing

TODO: Mention that this is from
https://github.com/samgd/repaper/blob/9c874f81eaffdc7e8096b46ee44b36b6dc6153b8/tests/utils/utils.py

"""
from typing import Callable, Optional

import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies as st
import numpy as np
import torch


def n_dims(
    min_dim_size: int = 1, max_dim_size: int = 5
) -> st.SearchStrategy[int]:
    """Returns a strategy which generates an integer dimenion size."""
    return st.integers(min_value=min_dim_size, max_value=max_dim_size)


def elements_of_type(
    dtype: np.dtype = np.float32, filter_: Optional[Callable] = None
) -> st.SearchStrategy:
    """Returns a strategy which generates elements of the given numpy dtype."""
    elems = None
    if dtype in (np.float16, np.float32, np.float64):
        elems = st.floats(min_value=-1.0, max_value=1.0)
    elif dtype is np.uint8:
        elems = st.integers(min_value=0, max_value=2 ** 8 - 1)
    elif dtype is np.uint32:
        elems = st.integers(min_value=0, max_value=2 ** 32 - 1)
    elif dtype is np.uint64:
        elems = st.integers(min_value=0, max_value=2 ** 64 - 1)
    elif dtype is np.int8:
        elems = st.integers(min_value=-2 ** 7, max_value=2 ** 7 - 1)
    elif dtype is np.int32:
        elems = st.integers(min_value=-2 ** 31, max_value=2 ** 31 - 1)
    elif dtype is np.int64:
        elems = st.integers(min_value=-2 ** 63, max_value=2 ** 63 - 1)
    elif dtype is np.bool:
        elems = st.booleans()
    else:
        raise ValueError("Unexpected dtype without elements provided")
    return elems if filter_ is None else elems.filter(filter_)


def arrays(
    shape: st.SearchStrategy,
    dtype: np.dtype = np.float32,
    elements: Optional[st.SearchStrategy] = None,
) -> st.SearchStrategy[np.ndarray]:
    """Returns a strategy for generating `np.ndarray`s."""
    if elements is None:
        elements = elements_of_type(dtype)
    return hypothesis.extra.numpy.arrays(dtype, shape=shape, elements=elements)


def tensors(
    min_n_dims: int = 0,
    max_n_dims: int = 4,
    dtype: np.dtype = np.float32,
    elements: Optional[st.SearchStrategy] = None,
    **kwargs
) -> st.SearchStrategy[torch.Tensor]:
    """Returns a strategy for generating `torch.Tensors`s of a given dtype."""
    shape = st.lists(n_dims(**kwargs), min_size=min_n_dims, max_size=max_n_dims)
    return shape.flatmap(
        lambda s: arrays(s, dtype, elements).map(lambda a: torch.tensor(a))
    )
