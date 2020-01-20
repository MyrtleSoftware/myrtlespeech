import warnings
from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from hypothesis import assume
from myrtlespeech.protos import dataset_pb2
from myrtlespeech.protos import range_pb2

from tests.data.test_alphabet import random_alphabet
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def datasets(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[dataset_pb2.Dataset],
    st.SearchStrategy[Tuple[dataset_pb2.Dataset, Dict]],
]:
    """Returns a SearchStrategy for a Dataset plus maybe the kwargs."""
    kwargs: Dict = {}

    desc = dataset_pb2.Dataset.DESCRIPTOR
    dataset_type_str = draw(
        st.sampled_from(
            [f.name for f in desc.oneofs_by_name["supported_datasets"].fields]
        )
    )

    # get kwargs for chosen dataset_type_str
    if dataset_type_str == "fake_speech_to_text":
        audio_ms_lower = draw(st.integers(1, 1000))
        audio_ms_upper = draw(st.integers(audio_ms_lower, 4 * audio_ms_lower))
        audio_ms = range_pb2.Range(lower=audio_ms_lower, upper=audio_ms_upper)

        label_symbols = "".join(draw(random_alphabet(min_size=2)).symbols)

        label_len_lower = draw(st.integers(1, 1000))
        label_len_upper = draw(
            st.integers(label_len_lower, 4 * label_len_lower)
        )
        label_len = range_pb2.Range(
            lower=label_len_lower, upper=label_len_upper
        )

        kwargs["fake_speech_to_text"] = dataset_pb2.Dataset.FakeSpeechToText(
            dataset_len=draw(st.integers(1, 100)),
            audio_ms=audio_ms,
            label_symbols=label_symbols,
            label_len=label_len,
        )
    elif dataset_type_str == "librispeech":
        warnings.warn("librispeech dataset not supported")
        assume(False)
    elif dataset_type_str == "commonvoice":
        warnings.warn("commonvoice dataset not supported")
        assume(False)
    elif dataset_type_str == "composite":
        warnings.warn("composite dataset not supported")
        assume(False)
    else:
        raise ValueError(
            f"test does not support generation of {dataset_type_str}"
        )

    # initialise dataset and return
    all_fields_set(dataset_pb2.Dataset, kwargs)
    dataset = dataset_pb2.Dataset(**kwargs)  # type: ignore
    if not return_kwargs:
        return dataset
    return dataset, kwargs
