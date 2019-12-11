from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import transducer_encoder_pb2

from tests.protos.test_fully_connected import fully_connecteds
from tests.protos.test_rnn import rnns
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def transducer_encoder(
    draw, time_reduction: bool, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[transducer_encoder_pb2.TransducerEncoder],
    st.SearchStrategy[Tuple[transducer_encoder_pb2.TransducerEncoder, Dict]],
]:
    """Returns a SearchStrategy for TransducerEncoder plus maybe the kwargs."""
    kwargs = {}
    to_ignore: List[str] = []
    kwargs["rnn1"] = draw(rnns())

    # maybe add time_reduction layer:
    if time_reduction:
        kwargs["time_reduction_factor"] = draw(st.integers(2, 3))
        kwargs["rnn2"] = draw(rnns())
    else:
        to_ignore = ["rnn2", "time_reduction_factor"]

    # maybe add fully connected layers at beginning and end of encoder:
    if draw(st.booleans()):
        kwargs["fc1"] = draw(fully_connecteds(valid_only=True))
        kwargs["fc2"] = draw(fully_connecteds(valid_only=True))
    else:
        to_ignore.extend(["fc1", "fc2"])

    all_fields_set(transducer_encoder_pb2.TransducerEncoder, kwargs, to_ignore)

    transducer_encoder = transducer_encoder_pb2.TransducerEncoder(**kwargs)
    if not return_kwargs:
        return transducer_encoder
    return transducer_encoder, kwargs
