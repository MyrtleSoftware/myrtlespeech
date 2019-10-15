from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import conv_layer_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def padding_modes(draw) -> st.SearchStrategy:
    """Returns a SearchStrategy over PADDING_MODEs."""
    return draw(st.sampled_from(list(conv_layer_pb2.PADDING_MODE.values())))


@st.composite
def conv1ds(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[conv_layer_pb2.Conv1d],
    st.SearchStrategy[Tuple[conv_layer_pb2.Conv1d, Dict]],
]:
    """Returns a SearchStrategy for a Conv1d layer plus maybe the kwargs."""
    kwargs = {}

    kwargs["output_channels"] = draw(st.integers(1, 32))
    kwargs["kernel_time"] = draw(st.integers(1, 7))
    kwargs["stride_time"] = draw(st.integers(1, 7))
    kwargs["padding_mode"] = draw(padding_modes())
    kwargs["bias"] = draw(st.booleans())

    all_fields_set(conv_layer_pb2.Conv1d, kwargs)
    conv1d = conv_layer_pb2.Conv1d(**kwargs)
    if not return_kwargs:
        return conv1d
    return conv1d, kwargs


@st.composite
def conv2ds(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[conv_layer_pb2.Conv2d],
    st.SearchStrategy[Tuple[conv_layer_pb2.Conv2d, Dict]],
]:
    """Returns a SearchStrategy for a Conv2d layer plus maybe the kwargs."""
    kwargs = {}

    kwargs["output_channels"] = draw(st.integers(1, 32))
    kwargs["kernel_time"] = draw(st.integers(1, 7))
    kwargs["kernel_feature"] = draw(st.integers(1, 7))
    kwargs["stride_time"] = draw(st.integers(1, 7))
    kwargs["stride_feature"] = draw(st.integers(1, 7))
    kwargs["padding_mode"] = draw(padding_modes())
    kwargs["bias"] = draw(st.booleans())

    all_fields_set(conv_layer_pb2.Conv2d, kwargs)
    conv2d = conv_layer_pb2.Conv2d(**kwargs)
    if not return_kwargs:
        return conv2d
    return conv2d, kwargs
