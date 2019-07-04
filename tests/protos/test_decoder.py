from typing import Union, Tuple, Dict

import hypothesis.strategies as st

from myrtlespeech.protos import decoder_pb2
from tests.protos.test_fully_connected import fully_connecteds
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def decoders(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[decoder_pb2.Decoder],
    st.SearchStrategy[Tuple[decoder_pb2.Decoder, Dict]],
]:
    """Returns a SearchStrategy for Decoder plus maybe the kwargs."""
    kwargs: Dict = {}

    # initialise oneof supported_decoders
    descript = decoder_pb2.Decoder.DESCRIPTOR
    supported_decoders = [
        f.name for f in descript.oneofs_by_name["supported_decoders"].fields
    ]
    decoder_str = draw(st.sampled_from(supported_decoders))
    if decoder_str == "fully_connected":
        kwargs["fully_connected"] = draw(fully_connecteds())
    else:
        raise ValueError(f"test does not support {decoder_str}")

    # initialise decoder and return
    all_fields_set(decoder_pb2.Decoder, kwargs)
    decoder = decoder_pb2.Decoder(**kwargs)
    if not return_kwargs:
        return decoder
    return decoder, kwargs
