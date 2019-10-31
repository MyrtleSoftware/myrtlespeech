from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import ctc_loss_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_t_losses(
    draw, return_kwargs: bool = False, alphabet_len: Optional[int] = None
) -> Union[
    st.SearchStrategy[ctc_loss_pb2.CTCLoss],
    st.SearchStrategy[Tuple[ctc_loss_pb2.CTCLoss, Dict]],
]:
    """Returns a SearchStrategy for RNNTLoss plus maybe the kwargs."""

    raise NotImplementedError()
    kwargs = {}

    end = 1000
    if alphabet_len is not None:
        end = max(0, alphabet_len - 1)
    kwargs["blank_index"] = draw(st.integers(0, end))

    kwargs["reduction"] = draw(
        st.sampled_from(ctc_loss_pb2.CTCLoss.REDUCTION.values())
    )

    all_fields_set(ctc_loss_pb2.CTCLoss, kwargs)
    ctc_loss = ctc_loss_pb2.CTCLoss(**kwargs)
    if not return_kwargs:
        return ctc_loss
    return ctc_loss, kwargs
