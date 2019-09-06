from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf.wrappers_pb2 import FloatValue
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
    to_ignore: List[str] = []

    kwargs["rnn_type"] = draw(st.sampled_from(rnn_pb2.RNN.RNN_TYPE.values()))
    kwargs["hidden_size"] = draw(st.integers(1, 32))
    kwargs["num_layers"] = draw(st.integers(1, 4))
    kwargs["bias"] = draw(st.booleans())
    kwargs["bidirectional"] = draw(st.booleans())
    if kwargs["rnn_type"] == rnn_pb2.RNN.RNN_TYPE.LSTM and kwargs["bias"]:
        kwargs["forget_gate_bias"] = FloatValue(
            value=draw(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)
            )
        )
    else:
        to_ignore = ["forget_gate_bias"]

    all_fields_set(rnn_pb2.RNN, kwargs, to_ignore)
    rnn = rnn_pb2.RNN(**kwargs)
    if not return_kwargs:
        return rnn
    return rnn, kwargs
