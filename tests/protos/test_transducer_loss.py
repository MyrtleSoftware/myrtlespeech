from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import transducer_loss_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def transducer_losses(
    draw, return_kwargs: bool = False, alphabet_len: Optional[int] = None
) -> Union[
    st.SearchStrategy[transducer_loss_pb2.TransducerLoss],
    st.SearchStrategy[Tuple[transducer_loss_pb2.TransducerLoss, Dict]],
]:
    """Returns a SearchStrategy for TransducerLoss plus maybe the kwargs."""

    kwargs = {}

    end = 1000
    if alphabet_len is not None:
        end = max(0, alphabet_len - 1)
    kwargs["blank_index"] = end

    kwargs["reduction"] = draw(
        st.sampled_from(transducer_loss_pb2.TransducerLoss.REDUCTION.values())
    )

    all_fields_set(transducer_loss_pb2.TransducerLoss, kwargs)
    transducer_loss = transducer_loss_pb2.TransducerLoss(**kwargs)
    if not return_kwargs:
        return transducer_loss
    return transducer_loss, kwargs
