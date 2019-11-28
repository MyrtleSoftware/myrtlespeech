from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf.wrappers_pb2 import FloatValue
from myrtlespeech.protos import lr_scheduler_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def constant_lrs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[lr_scheduler_pb2.ConstantLR],
    st.SearchStrategy[Tuple[lr_scheduler_pb2.ConstantLR, Dict]],
]:
    """Returns a SearchStrategy for an ConstantLR plus maybe the kwargs."""
    kwargs: Dict = {}

    # initialise and return
    all_fields_set(lr_scheduler_pb2.ConstantLR, kwargs)
    constant_lr = lr_scheduler_pb2.ConstantLR(**kwargs)
    if not return_kwargs:
        return constant_lr
    return constant_lr, kwargs


@st.composite
def step_lrs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[lr_scheduler_pb2.StepLR],
    st.SearchStrategy[Tuple[lr_scheduler_pb2.StepLR, Dict]],
]:
    """Returns a SearchStrategy for an StepLR plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["step_size"] = draw(st.integers(1, 30))

    kwargs["gamma"] = FloatValue(
        value=draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False))
    )

    # initialise and return
    all_fields_set(lr_scheduler_pb2.StepLR, kwargs)
    step_lr = lr_scheduler_pb2.StepLR(**kwargs)
    if not return_kwargs:
        return step_lr
    return step_lr, kwargs


@st.composite
def exponential_lrs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[lr_scheduler_pb2.ExponentialLR],
    st.SearchStrategy[Tuple[lr_scheduler_pb2.ExponentialLR, Dict]],
]:
    """Returns a SearchStrategy for an ExponentialLR plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["gamma"] = draw(
        st.floats(min_value=1e-4, max_value=1.0, allow_nan=False)
    )

    # initialise and return
    all_fields_set(lr_scheduler_pb2.ExponentialLR, kwargs)
    exponential_lr = lr_scheduler_pb2.ExponentialLR(**kwargs)
    if not return_kwargs:
        return exponential_lr
    return exponential_lr, kwargs


@st.composite
def cosine_annealing_lrs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[lr_scheduler_pb2.CosineAnnealingLR],
    st.SearchStrategy[Tuple[lr_scheduler_pb2.CosineAnnealingLR, Dict]],
]:
    """Returns a SearchStrategy for an CosineAnnealingLR plus maybe the
    kwargs. """
    kwargs: Dict = {}

    kwargs["t_max"] = draw(st.integers(1, 30))

    kwargs["eta_min"] = FloatValue(
        value=draw(st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False))
    )

    # initialise and return
    all_fields_set(lr_scheduler_pb2.CosineAnnealingLR, kwargs)
    cosine_annealing_lr = lr_scheduler_pb2.CosineAnnealingLR(**kwargs)
    if not return_kwargs:
        return cosine_annealing_lr
    return cosine_annealing_lr, kwargs
