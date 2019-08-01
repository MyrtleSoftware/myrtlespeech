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

    kwargs["momentum"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    )

    kwargs["l2_weight_decay"] = FloatValue(
        value=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    )

    kwargs["nesterov_momentum"] = draw(st.booleans())

    # initialise and return
    all_fields_set(optimizer_pb2.SGD, kwargs)
    sgd = optimizer_pb2.SGD(**kwargs)
    if not return_kwargs:
        return sgd
    return sgd, kwargs
