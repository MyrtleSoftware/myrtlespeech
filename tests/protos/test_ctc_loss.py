from typing import Union, Tuple, Dict

import hypothesis.strategies as st

from myrtlespeech.protos import ctc_loss_pb2
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def ctc_losses(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[ctc_loss_pb2.CTCLoss],
    st.SearchStrategy[Tuple[ctc_loss_pb2.CTCLoss, Dict]],
]:
    """Returns a SearchStrategy for CTCLoss plus maybe the kwargs."""
    kwargs = {}
    kwargs["blank_index"] = draw(st.integers(0, 1000))
    kwargs["reduction"] = draw(
        st.sampled_from(ctc_loss_pb2.CTCLoss.REDUCTION.values())
    )

    all_fields_set(ctc_loss_pb2.CTCLoss, kwargs)
    ctc_loss = ctc_loss_pb2.CTCLoss(**kwargs)
    if not return_kwargs:
        return ctc_loss
    return ctc_loss, kwargs
