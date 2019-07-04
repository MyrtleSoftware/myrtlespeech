from typing import Union, Tuple, Dict

import hypothesis.strategies as st

from myrtlespeech.protos import fully_connected_pb2
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def fully_connecteds(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[fully_connected_pb2.FullyConnected],
    st.SearchStrategy[Tuple[fully_connected_pb2.FullyConnected, Dict]],
]:
    """Returns a SearchStrategy for a FC layer plus maybe the kwargs."""
    kwargs = {}

    kwargs["num_hidden_layers"] = draw(st.integers(0, 3))
    kwargs["hidden_size"] = draw(st.integers(1, 32))
    kwargs["hidden_activation_fn"] = draw(
        st.sampled_from(
            fully_connected_pb2.FullyConnected.ACTIVATION_FN.values()
        )
    )

    all_fields_set(fully_connected_pb2.FullyConnected, kwargs)
    fc = fully_connected_pb2.FullyConnected(**kwargs)
    if not return_kwargs:
        return fc
    return fc, kwargs
