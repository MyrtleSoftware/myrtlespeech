import warnings
from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from hypothesis import assume
from myrtlespeech.protos import pre_process_step_pb2

from tests.protos.test_stage import stages
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def pre_process_steps(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.PreProcessStep],
    st.SearchStrategy[Tuple[pre_process_step_pb2.PreProcessStep, Dict]],
]:
    """Returns a SearchStrategy for a pre_process_step plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["stage"] = draw(stages())

    descript = pre_process_step_pb2.PreProcessStep.DESCRIPTOR
    step_type_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["pre_process_step"].fields]
        )
    )

    if step_type_str == "mfcc":
        kwargs["mfcc"] = draw(_mfccs())
    elif step_type_str == "standardize":
        warnings.warn("TODO")
        assume(False)
    else:
        raise ValueError(f"unknown pre_process_step type {step_type_str}")

    # initialise return
    all_fields_set(pre_process_step_pb2.PreProcessStep, kwargs)
    step = pre_process_step_pb2.PreProcessStep(**kwargs)
    if not return_kwargs:
        return step
    return step, kwargs


@st.composite
def _mfccs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.MFCC],
    st.SearchStrategy[Tuple[pre_process_step_pb2.MFCC, Dict]],
]:
    """Returns a SearchStrategy for MFCCs plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["numcep"] = draw(st.integers(1, 128))
    kwargs["winlen"] = draw(st.floats(0.01, 0.9))
    kwargs["winstep"] = draw(st.floats(0.01, 0.9))

    # initialise and return
    all_fields_set(pre_process_step_pb2.MFCC, kwargs)
    mfcc = pre_process_step_pb2.MFCC(**kwargs)  # type: ignore
    if not return_kwargs:
        return mfcc
    return mfcc, kwargs
