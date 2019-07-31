from typing import Union, Tuple, Dict

import hypothesis.strategies as st

from myrtlespeech.protos import encoder_pb2
from tests.protos.test_cnn_rnn_encoder import cnn_rnn_encoders
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def encoders(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[encoder_pb2.Encoder],
    st.SearchStrategy[Tuple[encoder_pb2.Encoder, Dict]],
]:
    """Returns a SearchStrategy for Encoder plus maybe the kwargs."""
    kwargs: Dict = {}

    # verify test can generate all "supported_encoders" and draw one
    des = encoder_pb2.Encoder.DESCRIPTOR
    encoder_type_str = draw(
        st.sampled_from(
            [f.name for f in des.oneofs_by_name["supported_encoders"].fields]
        )
    )

    # get kwargs for chosen encoder_type
    if encoder_type_str == "cnn_rnn_encoder":
        kwargs["cnn_rnn_encoder"] = draw(cnn_rnn_encoders())
    else:
        raise ValueError(
            f"test does not support generation of {encoder_type_str}"
        )

    # initialise encoder_type and return
    all_fields_set(encoder_pb2.Encoder, kwargs)
    encoder = encoder_pb2.Encoder(**kwargs)  # type: ignore
    if not return_kwargs:
        return encoder
    return encoder, kwargs
