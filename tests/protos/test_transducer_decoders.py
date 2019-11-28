from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import transducer_beam_decoder_pb2
from myrtlespeech.protos import transducer_greedy_decoder_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def transducer_beam_decoder(
    draw,
    return_kwargs: bool = False,
    alphabet_len: Optional[int] = None,
    blank_index: Optional[int] = None,
) -> Union[
    st.SearchStrategy[transducer_beam_decoder_pb2.TransducerBeamDecoder],
    st.SearchStrategy[
        Tuple[transducer_beam_decoder_pb2.TransducerBeamDecoder, Dict]
    ],
]:
    """Returns an st for TransducerBeamDecoder plus maybe the kwargs."""

    kwargs: Dict = {}

    end = 100
    if alphabet_len is not None:
        end = max(0, alphabet_len - 1)

    if blank_index is not None:
        kwargs["blank_index"] = blank_index
    else:
        kwargs["blank_index"] = draw(st.integers(0, end))

    kwargs["beam_width"] = draw(st.integers(1, 16))

    kwargs["length_norm"] = draw(st.booleans())

    kwargs["max_symbols_per_step"] = draw(st.integers(0, 4))

    # initialise and return
    all_fields_set(transducer_beam_decoder_pb2.TransducerBeamDecoder, kwargs)
    beam_decoder = transducer_beam_decoder_pb2.TransducerBeamDecoder(**kwargs)
    if not return_kwargs:
        return beam_decoder
    return beam_decoder, kwargs


@st.composite
def transducer_greedy_decoder(
    draw,
    return_kwargs: bool = False,
    alphabet_len: Optional[int] = None,
    blank_index: Optional[int] = None,
) -> Union[
    st.SearchStrategy[transducer_greedy_decoder_pb2.TransducerGreedyDecoder],
    st.SearchStrategy[
        Tuple[transducer_greedy_decoder_pb2.TransducerGreedyDecoder, Dict]
    ],
]:
    """Returns an st for TransducerGreedyDecoder plus maybe the kwargs."""
    kwargs: Dict = {}

    end = 100
    if alphabet_len is not None:
        end = max(0, alphabet_len - 1)

    if blank_index is not None:
        kwargs["blank_index"] = blank_index
    else:
        kwargs["blank_index"] = end

    kwargs["max_symbols_per_step"] = draw(st.integers(0, 4))

    # initialise and return
    all_fields_set(
        transducer_greedy_decoder_pb2.TransducerGreedyDecoder, kwargs
    )
    greedy_decoder = transducer_greedy_decoder_pb2.TransducerGreedyDecoder(
        **kwargs
    )
    if not return_kwargs:
        return greedy_decoder
    return greedy_decoder, kwargs
