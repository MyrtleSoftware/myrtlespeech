import pickle
from typing import Dict
from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from myrtlespeech.data.spec_augment import sparse_image_warp
from myrtlespeech.data.spec_augment import spec_augment


@st.composite
def spec_augements_args(
    draw, multimask: bool
) -> st.SearchStrategy[Tuple[torch.Tensor, Dict]]:
    """Returns a SearchStrategy for arguments for spec_augment"""
    # Choose kwargs
    kwargs: Dict = {}

    kwargs["time_warping_para"] = 0  # Don't test time_warping
    kwargs["frequency_masking_para"] = draw(st.integers(10, 100))
    kwargs["time_masking_para"] = draw(st.integers(10, 30))
    if multimask:
        kwargs["frequency_mask_num"] = draw(st.integers(1, 3))
        kwargs["time_mask_num"] = draw(st.integers(1, 3))
    else:
        kwargs["frequency_mask_num"] = 1
        kwargs["time_mask_num"] = 1

    # Choose 'spectogram' size and instantiate
    v = draw(st.integers(64, 256))
    tau = draw(st.integers(20, 50))
    spec = torch.ones(1, v, tau)
    return spec, kwargs


@given(args=spec_augements_args(False))
def test_mask_below_zero_upper_bound(args):
    spec, kwargs = args
    v = spec.shape[1]
    tau = spec.shape[2]
    freq_max = kwargs["frequency_masking_para"]
    time_max = kwargs["time_masking_para"]
    zeros_max = freq_max * tau + time_max * v - freq_max * time_max
    warped_spec = spec_augment(spec, **kwargs)
    assert torch.sum(warped_spec == 0) <= zeros_max


@given(args=spec_augements_args(True))
def test_masks_whole_freq(args):
    spec, kwargs = args
    warped_spec = spec_augment(spec, **kwargs)
    top_row = warped_spec[0, :, 0]
    masked_cols = torch.where(top_row == 0)[0]
    assert torch.all(warped_spec[0, masked_cols, :] == 0)


@given(args=spec_augements_args(True))
def test_masks_whole_time(args):
    spec, kwargs = args
    warped_spec = spec_augment(spec, **kwargs)
    top_col = warped_spec[0, 0, :]
    masked_rows = torch.where(top_col == 0)[0]
    assert torch.all(warped_spec[0, :, masked_rows] == 0)


@given(args=spec_augements_args(True))
def test_freqs_continous(args):
    spec, kwargs = args
    warped_spec = spec_augment(spec, **kwargs)
    top_row = warped_spec[0, :, 0]
    masked_cols = (torch.where(top_row == 0)[0]).numpy()
    assert sum(np.diff(masked_cols) != 1) <= kwargs["frequency_mask_num"] - 1


@given(args=spec_augements_args(True))
def test_time_continous(args):
    spec, kwargs = args
    warped_spec = spec_augment(spec, **kwargs)
    top_col = warped_spec[0, 0, :]
    masked_rows = (torch.where(top_col == 0)[0]).numpy()
    assert sum(np.diff(masked_rows) != 1) <= kwargs["time_mask_num"] - 1
