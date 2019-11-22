import warnings
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from hypothesis import assume
from myrtlespeech.builders.speech_to_text import _build_pre_process_steps
from myrtlespeech.protos import ctc_greedy_decoder_pb2
from myrtlespeech.protos import speech_to_text_pb2

from tests.data.test_alphabet import random_alphabet
from tests.protos.test_ctc_beam_decoder import ctc_beam_decoders
from tests.protos.test_ctc_loss import ctc_losses
from tests.protos.test_deep_speech_1 import deep_speech_1s
from tests.protos.test_deep_speech_2 import deep_speech_2s
from tests.protos.test_pre_process_step import pre_process_steps
from tests.protos.test_rnn_t import rnn_t
from tests.protos.test_rnn_t_decoders import rnn_t_beam_decoder
from tests.protos.test_rnn_t_decoders import rnn_t_greedy_decoder
from tests.protos.test_rnn_t_loss import rnn_t_losses
from tests.protos.utils import all_fields_set

# Fixtures and Strategies -----------------------------------------------------


@st.composite
def speech_to_texts(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[speech_to_text_pb2.SpeechToText],
    st.SearchStrategy[Tuple[speech_to_text_pb2.SpeechToText, Dict]],
]:
    """Returns a SearchStrategy for a SpeechToText model + maybe the kwargs."""
    kwargs: Dict = {}
    kwargs["alphabet"] = "".join(draw(random_alphabet(min_size=2)).symbols)

    descript = speech_to_text_pb2.SpeechToText.DESCRIPTOR

    # preprocess step
    kwargs["pre_process_step"] = []
    if draw(st.booleans()):
        kwargs["pre_process_step"].append(draw(pre_process_steps()))

    # record input_features and input_channels to ensure built model is valid
    _, input_features, input_channels = _build_pre_process_steps(
        kwargs["pre_process_step"]
    )

    # model
    model_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name["supported_models"].fields
            ]
        )
    )
    if model_str == "deep_speech_1":
        kwargs[model_str] = draw(deep_speech_1s())
    elif model_str == "deep_speech_2":
        kwargs[model_str] = draw(deep_speech_2s())
        warnings.warn(
            "TODO: fix hack that assumes input_features>200 for deep_speech_2"
        )
        assume(input_features > 200)
    elif model_str == "rnn_t":
        kwargs[model_str] = draw(rnn_t())
    else:
        raise ValueError(f"unknown model type {model_str}")

    # choice of two paths:
    # 1) ds2/ds1 -> `ctc`
    # 2) rnn_t -> `rnnt`

    # record blank index to share between CTC components
    blank_index: Optional[int] = None

    # loss
    loss_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name["supported_losses"].fields
            ]
        )
    )
    if loss_str == "ctc_loss":
        kwargs["ctc_loss"] = draw(
            ctc_losses(alphabet_len=len(kwargs["alphabet"]))
        )
        blank_index = kwargs["ctc_loss"].blank_index
    elif loss_str == "rnn_t_loss":
        kwargs["rnn_t_loss"] = draw(
            rnn_t_losses(alphabet_len=len(kwargs["alphabet"]))
        )
        blank_index = kwargs["rnn_t_loss"].blank_index
    else:
        raise ValueError(f"unknown loss type {loss_str}")

    # post process
    post_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name[
                    "supported_post_processes"
                ].fields
            ]
        )
    )
    if post_str == "ctc_greedy_decoder":
        if blank_index is None:
            blank_index = draw(
                st.integers(0, max(0, len(kwargs["alphabet"]) - 1))
            )
        kwargs["ctc_greedy_decoder"] = ctc_greedy_decoder_pb2.CTCGreedyDecoder(
            blank_index=blank_index
        )
    elif post_str == "ctc_beam_decoder":
        beam_kwargs = {"alphabet_len": len(kwargs["alphabet"])}
        if blank_index is not None:
            beam_kwargs["blank_index"] = blank_index
        kwargs["ctc_beam_decoder"] = draw(ctc_beam_decoders(**beam_kwargs))
    elif post_str == "rnn_t_beam_decoder":
        beam_kwargs = {"alphabet_len": len(kwargs["alphabet"])}
        if blank_index is not None:
            beam_kwargs["blank_index"] = blank_index
        kwargs["ctc_beam_decoder"] = draw(rnn_t_beam_decoder(**beam_kwargs))
    elif post_str == "rnn_t_greedy_decoder":
        beam_kwargs = {"alphabet_len": len(kwargs["alphabet"])}
        if blank_index is not None:
            beam_kwargs["blank_index"] = blank_index
        kwargs["ctc_beam_decoder"] = draw(rnn_t_greedy_decoder(**beam_kwargs))
    else:
        raise ValueError(f"unknown post_process type {post_str}")

    # initialise and return
    all_fields_set(speech_to_text_pb2.SpeechToText, kwargs)
    try:
        speech_to_text = speech_to_text_pb2.SpeechToText(  # type: ignore
            **kwargs
        )
    except TypeError:
        speech_to_text = None
        warnings.warn(
            "This test has (effectively) been disabled. TODO: prevent networks"
            f"that are trained with rnnt being built with"
            f"ctc-loss and vice versa"
        )

    if not return_kwargs:
        return speech_to_text
    return speech_to_text, kwargs
