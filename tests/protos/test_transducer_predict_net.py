from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import transducer_predict_net_pb2

from tests.protos.test_rnn import rnns
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def transducer_predict_net(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[transducer_predict_net_pb2.TransducerPredictNet],
    st.SearchStrategy[
        Tuple[transducer_predict_net_pb2.TransducerPredictNet, Dict]
    ],
]:
    """Returns a SearchStrategy for prediction net plus maybe the kwargs."""
    kwargs: Dict = {}
    kwargs["pred_nn"] = draw(_pred_nn())

    all_fields_set(transducer_predict_net_pb2.TransducerPredictNet, kwargs)

    pred_net = transducer_predict_net_pb2.TransducerPredictNet(**kwargs)
    if not return_kwargs:
        return pred_net
    return pred_net, kwargs


@st.composite
def _pred_nn(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[transducer_predict_net_pb2.PredNN],
    st.SearchStrategy[Tuple[transducer_predict_net_pb2.PredNN, Dict]],
]:
    """Returns a SearchStrategy for PredNN plus maybe the kwargs."""
    kwargs: Dict = {}
    kwargs["rnn"] = draw(rnns(batch_first=True))

    all_fields_set(transducer_predict_net_pb2.PredNN, kwargs)
    pred_nn = transducer_predict_net_pb2.PredNN(**kwargs)
    if not return_kwargs:
        return pred_nn
    return pred_nn, kwargs
