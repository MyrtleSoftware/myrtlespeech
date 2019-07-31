"""Based on Sam's Repaper `FakeDataset testing
<https://github.com/samgd/repaper/blob/c7332d96ac8b0db0d92ec2dbed63496a5ce0ed5f/tests/data/test_fake.py>`_.
"""
import warnings
from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from myrtlespeech.data.dataset.fake import FakeDataset
from myrtlespeech.data.dataset.fake import speech_to_text
from myrtlespeech.data.dataset.fake import SpeechToTextGen


# Setup/teardown --------------------------------------------------------------


def setup_module(module) -> None:
    """Initialise CUDA context for module.

    Why? PyTorch lazily initializes the CUDA context the first time it is used.
    Initializing the context is slow and, for tests that use CUDA, this results
    in test failures due to ``Unreliable test timings!``.

    This function forces the creation of the CUDA context before the tests
    begin by creating a :py:class:``torch.Tensor`` with device set to
    ``torch.device('cuda:0')``.
    """
    if torch.cuda.is_available():
        torch.tensor([1.0], device=torch.device("cuda:0"))


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def random_speech_to_text_kwargs(draw) -> st.SearchStrategy[Dict]:
    """Generates valid kwargs for ``speech_to_text``."""
    kwargs = {}

    a_lower = draw(st.integers(min_value=1, max_value=1000))
    a_upper = draw(st.integers(min_value=a_lower, max_value=2 * a_lower))
    kwargs["audio_ms"] = (a_lower, a_upper)  # type: ignore

    kwargs["label_symbols"] = list(  # type: ignore
        draw(st.sets(st.characters()))
    )
    l_lower = draw(st.integers(min_value=0, max_value=100))
    l_upper = draw(st.integers(min_value=l_lower, max_value=2 * l_lower))
    kwargs["label_len"] = (l_lower, l_upper)

    kwargs["audio_channels"] = draw(st.integers(min_value=1, max_value=4))

    dtypes = [torch.float64, torch.float32, torch.int32, torch.int16]

    kwargs["audio_dtype"] = draw(st.sampled_from(dtypes))

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    else:
        warnings.warn("CUDA not available", RuntimeWarning)
    kwargs["audio_device"] = draw(st.sampled_from(devices))

    if (
        kwargs["audio_device"] == torch.device("cpu")
        and torch.cuda.is_available()
    ):
        kwargs["audio_pin_memory"] = draw(st.booleans())
    else:
        kwargs["audio_pin_memory"] = False  # type: ignore

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
    kwargs["audio_sample_rate"] = draw(st.sampled_from(common_sample_rates))

    return kwargs


@st.composite
def random_speech_to_text(
    draw
) -> st.SearchStrategy[Tuple[SpeechToTextGen, Dict]]:
    """Generates different speech_to_text functions."""
    kwargs = draw(random_speech_to_text_kwargs())
    return speech_to_text(**kwargs), kwargs


@st.composite
def fake_dataset_strategy(
    draw, return_kwargs: bool = True
) -> Union[
    st.SearchStrategy[FakeDataset], st.SearchStrategy[Tuple[FakeDataset, Dict]]
]:
    """Defines a SearchStrategy for FakeDataset's."""
    generator, kwargs = draw(random_speech_to_text())
    dataset_len = draw(st.integers(min_value=0, max_value=10000))

    kwargs["generator"] = generator
    kwargs["dataset_len"] = dataset_len

    fake_dataset = FakeDataset(generator=generator, dataset_len=dataset_len)

    if return_kwargs:
        return fake_dataset, kwargs
    return fake_dataset


# Tests -----------------------------------------------------------------------

# speech_to_text ------------------------------------


@given(data=st.data(), kwargs=random_speech_to_text_kwargs())
def test_error_raised_when_audio_ms_lower_greater_than_upper(
    data, kwargs
) -> None:
    """Ensures ``ValueError`` raised when ``audio_ms[0] > audio_ms[1]``."""
    upper = kwargs["audio_ms"][1]
    invalid_lower = data.draw(st.integers(min_value=upper + 1))
    kwargs["audio_ms"] = (invalid_lower, upper)
    with pytest.raises(ValueError):
        speech_to_text(**kwargs)


@given(data=st.data(), kwargs=random_speech_to_text_kwargs())
def test_error_raised_when_audio_ms_less_than_one(data, kwargs) -> None:
    """Ensures ``ValueError`` raised when ``audio_ms[0] <= 0``."""
    invalid_lower = data.draw(st.integers(min_value=-1000, max_value=0))
    kwargs["audio_ms"] = (invalid_lower, kwargs["audio_ms"][1])
    with pytest.raises(ValueError):
        speech_to_text(**kwargs)


@given(data=st.data(), kwargs=random_speech_to_text_kwargs())
def test_error_raised_when_label_len_lower_greater_than_upper(
    data, kwargs
) -> None:
    """Ensures ``ValueError`` raised when ``label_len[0] > label_leb[1]``."""
    upper = kwargs["label_len"][1]
    invalid_lower = data.draw(st.integers(min_value=upper + 1))
    kwargs["label_len"] = (invalid_lower, upper)
    with pytest.raises(ValueError):
        speech_to_text(**kwargs)


@given(data=st.data(), kwargs=random_speech_to_text_kwargs())
def test_error_raised_when_label_len_less_than_zero(data, kwargs) -> None:
    """Ensures ``ValueError`` raised when ``label_len[0] < 0``."""
    invalid_lower = data.draw(st.integers(min_value=-1000, max_value=-1))
    kwargs["label_len"] = (invalid_lower, kwargs["label_len"][1])
    with pytest.raises(ValueError):
        speech_to_text(**kwargs)


@given(data=st.data(), kwargs=random_speech_to_text_kwargs())
def test_error_raised_when_audio_channels_less_than_one(data, kwargs) -> None:
    """Ensures ``ValueError`` raised when ``audio_channels < 1``."""
    kwargs["audio_channels"] = data.draw(
        st.integers(min_value=-1000, max_value=0)
    )
    with pytest.raises(ValueError):
        speech_to_text(**kwargs)


@given(data=st.data(), kwargs=random_speech_to_text_kwargs())
def test_error_raised_when_dtype_invalid(data, kwargs) -> None:
    """Ensures ``ValueError`` raised when ``audio_dtype`` invalid."""
    invalid_dtypes = [torch.float16, torch.uint8, torch.int8]
    kwargs["audio_dtype"] = data.draw(st.sampled_from(invalid_dtypes))
    with pytest.raises(ValueError):
        speech_to_text(**kwargs)


# FakeDataset ---------------------------------------


@given(fake_dataset_kwargs=fake_dataset_strategy())
def test_fake_dataset_len(fake_dataset_kwargs) -> None:
    """Ensures ``len(FakeDataset())`` is correct."""
    assert len(fake_dataset_kwargs[0]) == fake_dataset_kwargs[1]["dataset_len"]


@given(data=st.data(), fake_dataset=fake_dataset_strategy(return_kwargs=False))
def test_fake_dataset_valid_keys(data, fake_dataset) -> None:
    """Ensures valid ``__getitem__`` keys do not raise an error."""
    assume(len(fake_dataset) > 0)
    key = data.draw(
        st.integers(
            min_value=-len(fake_dataset), max_value=len(fake_dataset) - 1
        )
    )
    fake_dataset[key]


@given(data=st.data(), fake_dataset=fake_dataset_strategy(return_kwargs=False))
def test_fake_dataset_invalid_keys(data, fake_dataset) -> None:
    """Ensures invalid ``__getitem__`` keys raise an ``IndexError``."""
    dlen = len(fake_dataset)
    assume(dlen > 0)
    key = data.draw(
        st.one_of(
            st.integers(min_value=-2 * dlen, max_value=-dlen - 1),
            st.integers(min_value=dlen, max_value=2 * dlen),
        )
    )
    with pytest.raises(IndexError):
        fake_dataset[key]


@given(data=st.data(), fake_dataset=fake_dataset_strategy(return_kwargs=False))
def test_fake_dataset_key_access_return_same_element(
    data, fake_dataset
) -> None:
    """Ensures dataset[key] returns the same value each time it is executed."""
    assume(len(fake_dataset) > 0)
    key = data.draw(
        st.integers(
            min_value=-len(fake_dataset), max_value=len(fake_dataset) - 1
        )
    )
    a = fake_dataset[key]
    b = fake_dataset[key]

    try:
        assert a == b
    except RuntimeError:
        # Equality not supported
        pass


@given(fake_dataset=fake_dataset_strategy(return_kwargs=False))
def test_data_generation_changes_after_dataset_access(fake_dataset) -> None:
    """Ensures random data generated by PyTorch changes after dataset access.

    Why? The RandomDataset implementation used to (may still) temporarily
    modify PyTorch's random seed. Need to ensure that this is actually a
    temporary and not permanent modification else random data generation may
    become deterministic.
    """
    assume(len(fake_dataset) > 0)
    before = torch.empty([]).normal_()
    fake_dataset[0]
    after = torch.empty([]).normal_()

    assert before != after
