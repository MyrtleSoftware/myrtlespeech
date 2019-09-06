import warnings
from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import deep_speech_2_pb2

# from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def deep_speech_2s(
    draw, return_kwargs: bool = False, valid_only: bool = False
) -> Union[
    st.SearchStrategy[deep_speech_2_pb2.DeepSpeech2],
    st.SearchStrategy[Tuple[deep_speech_2_pb2.DeepSpeech2, Dict]],
]:
    """Returns a SearchStrategy for DeepSpeech2 plus maybe the kwargs."""
    warnings.warn("TODO")
    kwargs: Dict = {}

    # draw zero or more
    # kwargs["encoder"] = draw(encoders())
    # kwargs["decoder"] = draw(decoders(valid_only=valid_only))

    # initialise encoder and return
    # all_fields_set(encoder_decoder_pb2.EncoderDecoder, kwargs)
    # encoder_decoder = encoder_decoder_pb2.EncoderDecoder(  # type: ignore
    #     **kwargs
    # )
    deep_speech_2 = deep_speech_2_pb2.DeepSpeech2()

    if not return_kwargs:
        return deep_speech_2
    return deep_speech_2, kwargs
