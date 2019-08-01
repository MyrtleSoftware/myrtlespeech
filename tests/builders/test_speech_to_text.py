import warnings

from hypothesis import given
from myrtlespeech.builders.speech_to_text import build
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.model.speech_to_text import SpeechToText
from myrtlespeech.protos import speech_to_text_pb2

from tests.protos.test_speech_to_text import speech_to_texts


# Tests -----------------------------------------------------------------------


@given(stt_cfg=speech_to_texts())
def test_build_returns_speech_to_text(
    stt_cfg: speech_to_text_pb2.SpeechToText
) -> None:
    """Test that build returns a SpeechToText instance."""
    stt = build(stt_cfg)
    assert isinstance(stt, SpeechToText)
    assert isinstance(stt, SeqToSeq)
    warnings.warn("SpeechToText only built and not checked if correct")
