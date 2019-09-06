from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import vgg_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def vggs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[vgg_pb2.VGG], st.SearchStrategy[Tuple[vgg_pb2.VGG, Dict]]
]:
    """Returns a SearchStrategy for VGG plus maybe the kwargs."""
    kwargs = {}
    kwargs["vgg_config"] = draw(
        st.sampled_from(vgg_pb2.VGG.VGG_CONFIG.values())
    )
    kwargs["batch_norm"] = draw(st.booleans())
    kwargs["use_output_from_block"] = draw(
        st.sampled_from(vgg_pb2.VGG.USE_OUTPUT_FROM_BLOCK.values())
    )

    all_fields_set(vgg_pb2.VGG, kwargs)
    vgg = vgg_pb2.VGG(**kwargs)
    if not return_kwargs:
        return vgg
    return vgg, kwargs
