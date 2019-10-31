from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import rnn_t_loss_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_t_losses(
    draw, return_kwargs: bool = False, alphabet_len: Optional[int] = None
) -> Union[
    st.SearchStrategy[rnn_t_loss_pb2.RNNTLoss],
    st.SearchStrategy[Tuple[rnn_t_loss_pb2.RNNTLoss, Dict]],
]:
    """Returns a SearchStrategy for RNNTLoss plus maybe the kwargs."""

    kwargs = {}

    end = 1000
    if alphabet_len is not None:
        end = max(0, alphabet_len - 1)
    kwargs["blank_index"] = end

    kwargs["reduction"] = draw(
        st.sampled_from(rnn_t_loss_pb2.RNNTLoss.REDUCTION.values())
    )

    all_fields_set(rnn_t_loss_pb2.RNNTLoss, kwargs)
    rnn_t_loss = rnn_t_loss_pb2.RNNTLoss(**kwargs)
    if not return_kwargs:
        return rnn_t_loss
    return rnn_t_loss, kwargs
