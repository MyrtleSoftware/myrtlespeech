from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf import empty_pb2
from myrtlespeech.protos import activation_pb2
from myrtlespeech.protos import fully_connected_pb2

from tests.protos.test_activation import activations
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def fully_connecteds(
    draw, return_kwargs: bool = False, valid_only: bool = False
) -> Union[
    st.SearchStrategy[fully_connected_pb2.FullyConnected],
    st.SearchStrategy[Tuple[fully_connected_pb2.FullyConnected, Dict]],
]:
    """Returns a SearchStrategy for a FC layer plus maybe the kwargs."""
    eps = 1e-8

    kwargs = {}

    kwargs["num_hidden_layers"] = draw(st.integers(0, 3))
    if valid_only and kwargs["num_hidden_layers"] == 0:
        kwargs["hidden_size"] = None
        kwargs["activation"] = activation_pb2.Activation(
            identity=empty_pb2.Empty()
        )
        kwargs["dropout"] = 0
    else:
        kwargs["hidden_size"] = draw(st.integers(1, 32))
        kwargs["activation"] = draw(activations())
        if draw(st.booleans()):
            kwargs["dropout"] = draw(st.floats(eps, 1 - eps))
        else:
            kwargs["dropout"] = 0

    all_fields_set(fully_connected_pb2.FullyConnected, kwargs)
    fc = fully_connected_pb2.FullyConnected(**kwargs)
    if not return_kwargs:
        return fc
    return fc, kwargs
