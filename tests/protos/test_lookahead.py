from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import lookahead_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def lookaheads(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[lookahead_pb2.Lookahead],
    st.SearchStrategy[Tuple[lookahead_pb2.Lookahead, Dict]],
]:
    """Returns a SearchStrategy for a lookahead layer plus maybe the kwargs."""
    kwargs = {}
    kwargs["context"] = draw(st.integers(1, 32))

    all_fields_set(lookahead_pb2.Lookahead, kwargs)
    lookahead = lookahead_pb2.Lookahead(**kwargs)
    if not return_kwargs:
        return lookahead
    return lookahead, kwargs
