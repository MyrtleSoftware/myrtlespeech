from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf import empty_pb2
from myrtlespeech.protos import activation_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def activations(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[activation_pb2.Activation],
    st.SearchStrategy[Tuple[activation_pb2.Activation, Dict]],
]:
    """Returns a SearchStrategy for activation fns plus maybe the kwargs."""
    kwargs = {}

    descript = activation_pb2.Activation.DESCRIPTOR
    activation_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["activation"].fields]
        )
    )

    if activation_str == "identity":
        kwargs["identity"] = empty_pb2.Empty()
    elif activation_str == "hardtanh":
        kwargs["hardtanh"] = activation_pb2.Activation.Hardtanh(
            min_val=draw(st.floats(-20.0, -0.1, allow_nan=False)),
            max_val=draw(st.floats(0.1, 20.0, allow_nan=False)),
        )
    elif activation_str == "relu":
        kwargs["relu"] = activation_pb2.Activation.ReLU()
    else:
        raise ValueError(f"test does not support activation={activation_str}")

    all_fields_set(activation_pb2.Activation, kwargs)
    act = activation_pb2.Activation(**kwargs)
    if not return_kwargs:
        return act
    return act, kwargs
