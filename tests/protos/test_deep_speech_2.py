from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf import empty_pb2
from myrtlespeech.protos import conv_layer_pb2
from myrtlespeech.protos import deep_speech_2_pb2
from myrtlespeech.protos import lookahead_pb2

from tests.protos.test_activation import activations
from tests.protos.test_conv_layer import conv1ds
from tests.protos.test_conv_layer import conv2ds
from tests.protos.test_fully_connected import fully_connecteds
from tests.protos.test_lookahead import lookaheads
from tests.protos.test_rnn import rnns
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def deep_speech_2s(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[deep_speech_2_pb2.DeepSpeech2],
    st.SearchStrategy[Tuple[deep_speech_2_pb2.DeepSpeech2, Dict]],
]:
    """Returns a SearchStrategy for DeepSpeech2 plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["conv_block"] = draw(_conv_blocks())
    kwargs["rnn"] = draw(rnns())
    kwargs["lookahead_block"] = draw(_lookahead_blocks())
    kwargs["fully_connected"] = draw(fully_connecteds(valid_only=True))

    # initialise and return
    all_fields_set(deep_speech_2_pb2.DeepSpeech2, kwargs)
    ds2 = deep_speech_2_pb2.DeepSpeech2(**kwargs)  # type: ignore

    if not return_kwargs:
        return ds2
    return ds2, kwargs


@st.composite
def _conv_blocks(
    draw
) -> st.SearchStrategy[List[deep_speech_2_pb2.DeepSpeech2.ConvBlock]]:
    """Returns a SearchStrategy for ConvBlocks."""

    @st.composite
    def _blocks(draw):
        kwargs = {}

        convnd = draw(st.one_of(conv1ds(), conv2ds()))
        if isinstance(convnd, conv_layer_pb2.Conv1d):
            kwargs["conv1d"] = convnd
        else:
            kwargs["conv2d"] = convnd

        kwargs["batch_norm"] = draw(st.booleans())
        kwargs["activation"] = draw(activations())

        return deep_speech_2_pb2.DeepSpeech2.ConvBlock(**kwargs)

    return draw(st.lists(elements=_blocks(), min_size=1, max_size=5))


@st.composite
def _lookahead_blocks(
    draw
) -> st.SearchStrategy[deep_speech_2_pb2.DeepSpeech2.LookaheadBlock]:
    """Returns a SearchStrategy for a LookaheadBlock."""
    kwargs = {}

    lookahead = draw(st.one_of(lookaheads(), st.just(empty_pb2.Empty())))
    if isinstance(lookahead, lookahead_pb2.Lookahead):
        kwargs["lookahead"] = lookahead
    else:
        kwargs["no_lookahead"] = lookahead

    kwargs["activation"] = draw(activations())
    return deep_speech_2_pb2.DeepSpeech2.LookaheadBlock(**kwargs)
