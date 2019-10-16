from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import deep_speech_1_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def deep_speech_1s(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[deep_speech_1_pb2.DeepSpeech1],
    st.SearchStrategy[Tuple[deep_speech_1_pb2.DeepSpeech1, Dict]],
]:
    """Returns a SearchStrategy for DeepSpeech1 plus maybe the kwargs."""
    kwargs: Dict = {}

    # draw zero or more
    kwargs["n_hidden"] = draw(st.integers(1, 128))
    kwargs["drop_prob"] = draw(st.floats(0.0, 1.0, allow_nan=False))
    kwargs["relu_clip"] = draw(st.floats(1.0, 20.0))
    kwargs["forget_gate_bias"] = draw(st.floats(0.0, 1.0))

    # initialise and return
    all_fields_set(deep_speech_1_pb2.DeepSpeech1, kwargs)
    ds1 = deep_speech_1_pb2.DeepSpeech1(**kwargs)  # type: ignore

    if not return_kwargs:
        return ds1
    return ds1, kwargs
