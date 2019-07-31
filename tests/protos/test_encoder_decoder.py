from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import encoder_decoder_pb2

from tests.protos.test_decoder import decoders
from tests.protos.test_encoder import encoders
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def encoder_decoders(
    draw, return_kwargs: bool = False, valid_only: bool = False
) -> Union[
    st.SearchStrategy[encoder_decoder_pb2.EncoderDecoder],
    st.SearchStrategy[Tuple[encoder_decoder_pb2.EncoderDecoder, Dict]],
]:
    """Returns a SearchStrategy for EncoderDecoder plus maybe the kwargs."""
    kwargs = {}
    kwargs["encoder"] = draw(encoders())
    kwargs["decoder"] = draw(decoders(valid_only=valid_only))

    # initialise encoder and return
    all_fields_set(encoder_decoder_pb2.EncoderDecoder, kwargs)
    encoder_decoder = encoder_decoder_pb2.EncoderDecoder(  # type: ignore
        **kwargs
    )
    if not return_kwargs:
        return encoder_decoder
    return encoder_decoder, kwargs
