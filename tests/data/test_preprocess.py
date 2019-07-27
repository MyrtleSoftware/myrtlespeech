import decimal
import math
import warnings
from typing import Dict, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import note, given
from myrtlespeech.data.preprocess import MFCC, AddSequenceLength
from tests.utils.utils import tensors


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def random_mfcc_kwargs(draw) -> st.SearchStrategy[Dict]:
    """Generates valid kwargs for ``MFCC``."""
    kwargs = {}

    kwargs["numcep"] = draw(st.integers(min_value=1, max_value=240))
    kwargs["winlen"] = draw(
        st.floats(min_value=0.001, max_value=0.1, allow_nan=False)
    )
    kwargs["winstep"] = draw(
        st.floats(min_value=0.001, max_value=0.1, allow_nan=False)
    )
    common_sample_rates = [
        8000,
        11025,
        16000,
        22050,
        32000,
        37800,
        44056,
        44100,
        47250,
        48000,
        50000,
        50400,
        64000,
        88200,
        96000,
        176400,
        192000,
        352800,
    ]
    kwargs["sample_rate"] = draw(st.sampled_from(common_sample_rates))

    return kwargs


@st.composite
def random_mfcc(
    draw, return_kwargs: bool = True
) -> st.SearchStrategy[Tuple[MFCC, Dict]]:
    """Generates different ``MFCC`` ``Callable``s."""
    kwargs = draw(random_mfcc_kwargs())
    if return_kwargs:
        return MFCC(**kwargs), kwargs
    return MFCC(**kwargs)


# Tests -----------------------------------------------------------------------

# MFCC ----------------------------------------------


@given(data=st.data(), mfcc=random_mfcc(return_kwargs=False))
def test_MFCC_output_size_correct(data, mfcc):
    """Ensures MFCC output size matches expected value."""
    audio_len = data.draw(st.integers(min_value=1, max_value=10000))
    note(f"audio_len={audio_len}")

    audiodata = torch.empty([audio_len]).normal_()

    expected_mfcc_len = expected_len(
        audio_len, mfcc.winlen, mfcc.winstep, mfcc.sample_rate
    )
    expected_mfcc_features = mfcc.numcep

    with warnings.catch_warnings():
        # generated mfcc paramter combinations may be odd and cause warnings
        warnings.simplefilter("ignore")
        actual_mfcc_features, actual_mfcc_len = mfcc(audiodata).size()

    assert expected_mfcc_len == actual_mfcc_len
    assert expected_mfcc_features == actual_mfcc_features


def expected_len(audio_len, winlen, winstep, sample_rate):
    """Returns the expected length after applying a sliding window.

    Based on the `code
    <https://github.com/jameslyons/python_speech_features/blob/40c590269b57c64a8c1f1ddaaff2162008d1850c/python_speech_features/sigproc.py#L31>`_
    originally used to compute the MFCCs.
    """

    def round_half_up(number):
        return int(
            decimal.Decimal(number).quantize(
                decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP
            )
        )

    frame_len = int(round_half_up(winlen * sample_rate))
    frame_step = int(round_half_up(winstep * sample_rate))
    if audio_len <= frame_len:
        numframes = 1
    else:
        numframes = 1 + math.ceil((audio_len - frame_len) / frame_step)

    return numframes


# AddSequenceLength ---------------------------------


@given(data=st.data(), tensor=tensors(min_n_dims=1))
def test_add_sequence_length_returns_correct_seq_len(
    data, tensor: torch.Tensor
) -> None:
    """Ensures AddSequenceLength returns correct sequence length."""
    length_dim = data.draw(
        st.integers(min_value=0, max_value=len(tensor.size()) - 1)
    )

    add_seq_len = AddSequenceLength(length_dim=length_dim)

    out, seq_len = add_seq_len(tensor)

    assert torch.all(out == tensor)
    assert seq_len == torch.tensor([tensor.size(length_dim)])
