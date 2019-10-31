from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf.wrappers_pb2 import FloatValue
from myrtlespeech.protos import rnn_t_encoder_pb2

from tests.protos.test_rnn import rnns
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_t_encoder(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[rnn_t_encoder_pb2.RNNTEncoder],
    st.SearchStrategy[Tuple[rnn_t_encoder_pb2.RNNTEncoder, Dict]],
]:
    """Returns a SearchStrategy for RNNTEncoder plus maybe the kwargs."""
    kwargs = {}
    to_ignore: List[str] = []
    kwargs["rnn1"] = draw(rnns())
    if draw(st.booleans()):
        kwargs["time_reduction_factor"] = draw(st.integers(2, 3))
        kwargs["rnn2"] = draw(rnns(batch_first=False))
    else:
        to_ignore = ["rnn2", "time_reduction_factor"]

    all_fields_set(rnn_t_encoder_pb2.RNNTEncoder, kwargs, to_ignore)

    rnn_t_encoder = rnn_t_encoder_pb2.RNNTEncoder(**kwargs)
    if not return_kwargs:
        return rnn_t_encoder
    return rnn_t_encoder, kwargs
