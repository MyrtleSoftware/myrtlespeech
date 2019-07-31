from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf import empty_pb2
from myrtlespeech.protos import language_model_pb2

from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def language_models(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[Union[empty_pb2.Empty, Callable[[List[int]], float]]],
    st.SearchStrategy[
        Tuple[Union[empty_pb2.Empty, Callable[[List[int]], float]], Dict]
    ],
]:
    """Returns a SearchStrategy for language models plus maybe the kwargs."""
    kwargs: Dict = {}

    # initialise oneof supported_lms
    supported_lm = draw(_supported_lms())
    if isinstance(supported_lm, empty_pb2.Empty):
        kwargs["no_lm"] = supported_lm
    else:
        raise ValueError(f"unknown lm type {type(supported_lm)}")

    # initialise language model and return
    all_fields_set(language_model_pb2.LanguageModel, kwargs)
    lm = language_model_pb2.LanguageModel(**kwargs)
    if not return_kwargs:
        return lm
    return lm, kwargs


@st.composite
def _supported_lms(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[empty_pb2.Empty],
    st.SearchStrategy[Tuple[empty_pb2.Empty, Dict]],
]:
    """Returns a SearchStrategy for supported_lms plus maybe the kwargs."""
    kwargs: Dict = {}

    descript = language_model_pb2.LanguageModel.DESCRIPTOR
    lm_type_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["supported_lms"].fields]
        )
    )

    # get kwargs for chosen lm_type
    if lm_type_str == "no_lm":
        lm_type = empty_pb2.Empty
    else:
        raise ValueError(f"test does not support generation of {lm_type}")

    # initialise lm_type and return
    all_fields_set(lm_type, kwargs)
    lm = lm_type(**kwargs)  # type: ignore
    if not return_kwargs:
        return lm
    return lm, kwargs
