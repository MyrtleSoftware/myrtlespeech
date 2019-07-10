from typing import Union, Tuple, Dict

import hypothesis.strategies as st
from google.protobuf import empty_pb2
from google.protobuf.wrappers_pb2 import FloatValue, UInt32Value

from myrtlespeech.protos import ctc_beam_decoder_pb2
from tests.protos.test_language_model import language_models
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def ctc_beam_decoders(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[ctc_beam_decoder_pb2.CTCBeamDecoder],
    st.SearchStrategy[Tuple[ctc_beam_decoder_pb2.CTCBeamDecoder, Dict]],
]:
    """Returns a SearchStrategy for CTCBeamDecoder plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["blank_index"] = draw(st.integers(0, 100))
    kwargs["beam_width"] = draw(st.integers(1, 2048))
    kwargs["prune_threshold"] = draw(st.floats(0.0, 1.0, allow_nan=False))

    kwargs["language_model"] = draw(language_models())
    if not isinstance(kwargs["language_model"], empty_pb2.Empty):
        kwargs["lm_weight"] = FloatValue(
            value=draw(st.floats(allow_nan=False, allow_infinity=False))
        )

    kwargs["separator_index"] = UInt32Value(value=draw(st.integers(0, 100)))
    kwargs["word_weight"] = draw(
        st.floats(allow_nan=False, allow_infinity=False)
    )

    # initialise and return
    all_fields_set(ctc_beam_decoder_pb2.CTCBeamDecoder, kwargs)
    ctc_beam_decoder = ctc_beam_decoder_pb2.CTCBeamDecoder(**kwargs)
    if not return_kwargs:
        return ctc_beam_decoder
    return ctc_beam_decoder, kwargs
