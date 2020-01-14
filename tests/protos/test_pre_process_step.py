from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
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
    """Returns a SearchStrategy for a pre_process_step + maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["stage"] = draw(stages())

    descript = pre_process_step_pb2.PreProcessStep.DESCRIPTOR
    step_type_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name["pre_process_step"].fields
            ]
        )
    )

    if step_type_str == "mfcc":
        kwargs["mfcc"] = draw(_mfccs())
    elif step_type_str == "log_mel_filter_bank":
        kwargs["log_mel_filter_bank"] = draw(_log_mel_filter_banks())
    elif step_type_str == "spec_augment":
        kwargs["spec_augment"] = draw(_spec_augments())
    elif step_type_str == "standardize":
        kwargs["standardize"] = draw(_standardizes())
    elif step_type_str == "context_frames":
        kwargs["context_frames"] = draw(_context_frames())
    elif step_type_str == "downsample":
        kwargs["downsample"] = draw(_downsample())
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

    kwargs["n_mfcc"] = draw(st.integers(1, 128))
    kwargs["win_length"] = draw(st.integers(100, 400))
    kwargs["hop_length"] = draw(st.integers(50, kwargs["win_length"]))

    # initialise and return
    all_fields_set(pre_process_step_pb2.MFCC, kwargs)
    mfcc = pre_process_step_pb2.MFCC(**kwargs)  # type: ignore
    if not return_kwargs:
        return mfcc
    return mfcc, kwargs


@st.composite
def _spec_augments(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.SpecAugment],
    st.SearchStrategy[Tuple[pre_process_step_pb2.SpecAugment, Dict]],
]:
    """Returns a SearchStrategy for SpecAugments plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["feature_mask"] = draw(st.integers(0, 80))
    kwargs["time_mask"] = draw(st.integers(0, 100))
    kwargs["n_feature_masks"] = draw(st.integers(0, 3))
    kwargs["n_time_masks"] = draw(st.integers(0, 3))

    # initialise and return
    all_fields_set(pre_process_step_pb2.SpecAugment, kwargs)
    spec_augment = pre_process_step_pb2.SpecAugment(**kwargs)  # type: ignore
    if not return_kwargs:
        return spec_augment
    return spec_augment, kwargs


@st.composite
def _log_mel_filter_banks(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.LogMelFB],
    st.SearchStrategy[Tuple[pre_process_step_pb2.LogMelFB, Dict]],
]:
    r"""Returns a SearchStrategy for LogMelFBs plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["n_mels"] = draw(st.integers(1, 128))
    kwargs["win_length"] = draw(st.integers(100, 400))
    kwargs["hop_length"] = draw(st.integers(50, kwargs["win_length"]))

    # initialise and return
    all_fields_set(pre_process_step_pb2.LogMelFB, kwargs)
    lmfb = pre_process_step_pb2.LogMelFB(**kwargs)  # type: ignore
    if not return_kwargs:
        return lmfb
    return lmfb, kwargs


@st.composite
def _standardizes(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.Standardize],
    st.SearchStrategy[Tuple[pre_process_step_pb2.Standardize, Dict]],
]:
    """Returns a SearchStrategy for Standardizes plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["norm_type"] = draw(
        st.sampled_from(pre_process_step_pb2.Standardize.NormType.values())
    )

    # initialise and return
    all_fields_set(pre_process_step_pb2.Standardize, kwargs)
    std = pre_process_step_pb2.Standardize(**kwargs)  # type: ignore
    if not return_kwargs:
        return std
    return std, kwargs


@st.composite
def _context_frames(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.ContextFrames],
    st.SearchStrategy[Tuple[pre_process_step_pb2.ContextFrames, Dict]],
]:
    """Returns a SearchStrategy for ContextFrames plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["n_context"] = draw(st.integers(1, 18))

    # initialise and return
    all_fields_set(pre_process_step_pb2.ContextFrames, kwargs)
    cf = pre_process_step_pb2.ContextFrames(**kwargs)  # type: ignore
    if not return_kwargs:
        return cf
    return cf, kwargs


@st.composite
def _downsample(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[pre_process_step_pb2.Downsample],
    st.SearchStrategy[Tuple[pre_process_step_pb2.Downsample, Dict]],
]:
    """Returns a SearchStrategy for Downsample plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["subsample"] = draw(st.integers(2, 3))

    # initialise and return
    all_fields_set(pre_process_step_pb2.Downsample, kwargs)
    downsample = pre_process_step_pb2.Downsample(**kwargs)  # type: ignore
    if not return_kwargs:
        return downsample
    return downsample, kwargs
