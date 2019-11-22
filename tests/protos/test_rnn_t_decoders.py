from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import rnn_t_beam_decoder_pb2
from myrtlespeech.protos import rnn_t_greedy_decoder_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def rnn_t_beam_decoder(
    draw,
    return_kwargs: bool = False,
    alphabet_len: Optional[int] = None,
    blank_index: Optional[int] = None,
) -> Union[
    st.SearchStrategy[rnn_t_beam_decoder_pb2.RNNTBeamDecoder],
    st.SearchStrategy[Tuple[rnn_t_beam_decoder_pb2.RNNTBeamDecoder, Dict]],
]:
    """Returns a SearchStrategy for RNNTBeamDecoder plus maybe the kwargs."""

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
    all_fields_set(rnn_t_beam_decoder_pb2.RNNTBeamDecoder, kwargs)
    beam_decoder = rnn_t_beam_decoder_pb2.RNNTBeamDecoder(**kwargs)
    if not return_kwargs:
        return beam_decoder
    return beam_decoder, kwargs


@st.composite
def rnn_t_greedy_decoder(
    draw,
    return_kwargs: bool = False,
    alphabet_len: Optional[int] = None,
    blank_index: Optional[int] = None,
) -> Union[
    st.SearchStrategy[rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder],
    st.SearchStrategy[Tuple[rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder, Dict]],
]:
    """Returns a SearchStrategy for RNNTGreedyDecoder plus maybe the kwargs."""
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
    all_fields_set(rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder, kwargs)
    greedy_decoder = rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder(**kwargs)
    if not return_kwargs:
        return greedy_decoder
    return greedy_decoder, kwargs
