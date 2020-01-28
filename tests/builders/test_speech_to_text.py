import warnings

import pytest
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.speech_to_text import build
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.model.speech_to_text import SpeechToText
from myrtlespeech.protos import speech_to_text_pb2

from tests.protos.test_speech_to_text import speech_to_texts


# Tests -----------------------------------------------------------------------


@given(stt_cfg=speech_to_texts())
def test_build_returns_speech_to_text(
    stt_cfg: speech_to_text_pb2.SpeechToText,
) -> None:
    """Test that build returns a SpeechToText instance."""
    stt = build(stt_cfg)
    assert isinstance(stt, SpeechToText)
    assert isinstance(stt, SeqToSeq)
    warnings.warn("SpeechToText only built and not checked if correct")


@given(stt_cfg=speech_to_texts(valid_only=False))
def test_build_raises_value_error_when_model_or_post_process_invalid(
    stt_cfg: speech_to_text_pb2.SpeechToText,
) -> None:
    """Test build raises ValueError for invalid loss and model/post_process."""
    loss_type = stt_cfg.WhichOneof("supported_losses")
    model_type = stt_cfg.WhichOneof("supported_models")
    post_process_type = stt_cfg.WhichOneof("supported_post_processes")
    if loss_type == "ctc_loss":
        assume("transducer" in model_type or "transducer" in post_process_type)
    elif loss_type == "transducer_loss":
        assume(
            "transducer" not in model_type
            or "transducer" not in post_process_type
        )
    else:
        raise ValueError(f"loss_type={loss_type} not recognized")

    with pytest.raises(ValueError):
        build(stt_cfg)
