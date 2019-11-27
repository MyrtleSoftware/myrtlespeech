from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import transducer_pb2

from tests.protos.test_fully_connected import fully_connecteds
from tests.protos.test_transducer_encoder import transducer_encoder
from tests.protos.test_transducer_predict_net import transducer_predict_net
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def transducer(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[transducer_pb2.Transducer],
    st.SearchStrategy[Tuple[transducer_pb2.Transducer, Dict]],
]:
    """Returns a SearchStrategy for Transducer plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["transducer_encoder"], enc_kwargs = draw(
        transducer_encoder(return_kwargs=True)
    )
    kwargs["transducer_predict_net"], dec_kwargs = draw(
        transducer_predict_net(return_kwargs=True)
    )
    kwargs["joint_net"], fc_kwargs = draw(
        fully_connecteds(valid_only=True, return_kwargs=True)
    )

    # initialise and return
    all_fields_set(transducer_pb2.Transducer, kwargs)
    transducer = transducer_pb2.Transducer(**kwargs)  # type: ignore

    if not return_kwargs:
        return transducer
    return transducer, kwargs
