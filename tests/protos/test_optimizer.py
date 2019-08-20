from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf.wrappers_pb2 import FloatValue
from myrtlespeech.protos import optimizer_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def sgds(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[optimizer_pb2.SGD],
    st.SearchStrategy[Tuple[optimizer_pb2.SGD, Dict]],
]:
    """Returns a SearchStrategy for an SGD plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["learning_rate"] = draw(
        st.floats(min_value=1e-4, max_value=1.0, allow_nan=False)
    )

    kwargs["nesterov_momentum"] = draw(st.booleans())

    # nesterov momentum requires momentum > 0
    min_momentum = 0.0 if not kwargs["nesterov_momentum"] else 0.1
    kwargs["momentum"] = FloatValue(
        value=draw(
            st.floats(min_value=min_momentum, max_value=10.0, allow_nan=False)
        )
    )

    kwargs["l2_weight_decay"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    )

    # initialise and return
    all_fields_set(optimizer_pb2.SGD, kwargs)
    sgd = optimizer_pb2.SGD(**kwargs)
    if not return_kwargs:
        return sgd
    return sgd, kwargs


@st.composite
def adams(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[optimizer_pb2.Adam],
    st.SearchStrategy[Tuple[optimizer_pb2.Adam, Dict]],
]:
    """Returns a SearchStrategy for an Adam optimizer plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["learning_rate"] = draw(
        st.floats(min_value=1e-4, max_value=1.0, allow_nan=False)
    )

    kwargs["beta_1"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    )

    kwargs["beta_2"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    )

    kwargs["eps"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=1e-3, allow_nan=False))
    )

    kwargs["l2_weight_decay"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    )

    kwargs["amsgrad"] = draw(st.booleans())

    # initialise and return
    all_fields_set(optimizer_pb2.Adam, kwargs)
    adam = optimizer_pb2.Adam(**kwargs)
    if not return_kwargs:
        return adam
    return adam, kwargs
