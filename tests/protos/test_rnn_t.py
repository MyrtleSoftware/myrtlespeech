from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf import empty_pb2
from myrtlespeech.protos import conv_layer_pb2
from myrtlespeech.protos import lookahead_pb2
from myrtlespeech.protos import rnn_t_pb2

from tests.protos.test_fully_connected import fully_connecteds
from tests.protos.test_rnn import rnns
from tests.protos.test_rnn_t_encoder import rnn_t_encoder
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_t(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[rnn_t_pb2.RNNT],
    st.SearchStrategy[Tuple[rnn_t_pb2.RNNT, Dict]],
]:
    """Returns a SearchStrategy for RNNT plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["rnn_t_encoder"], enc_kwargs = draw(
        rnn_t_encoder(return_kwargs=True)
    )
    kwargs["dec_rnn"], dec_kwargs = draw(
        rnns(batch_first=True, return_kwargs=True)
    )
    kwargs["fully_connected"], fc_kwargs = draw(
        fully_connecteds(valid_only=True, return_kwargs=True)
    )

    # initialise and return
    all_fields_set(rnn_t_pb2.RNNT, kwargs)
    rnn_t = rnn_t_pb2.RNNT(**kwargs)  # type: ignore

    if not return_kwargs:
        return rnn_t
    return rnn_t, kwargs
