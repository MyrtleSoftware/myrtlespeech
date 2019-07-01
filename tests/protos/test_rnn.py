from typing import Union, Tuple, Dict

import hypothesis.strategies as st

from myrtlespeech.protos import rnn_pb2
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnns(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[rnn_pb2.RNN], st.SearchStrategy[Tuple[rnn_pb2.RNN, Dict]]
]:
    """Returns a SearchStrategy for RNN plus maybe the kwargs."""
    kwargs = {}

    kwargs["rnn_type"] = draw(st.sampled_from(rnn_pb2.RNN.RNN_TYPE.values()))
    kwargs["hidden_size"] = draw(st.integers(1, 32))
    kwargs["num_layers"] = draw(st.integers(1, 4))
    kwargs["bias"] = draw(st.booleans())
    kwargs["bidirectional"] = draw(st.booleans())

    all_fields_set(rnn_pb2.RNN, kwargs)
    rnn = rnn_pb2.RNN(**kwargs)
    if not return_kwargs:
        return rnn
    return rnn, kwargs
