import warnings

from hypothesis import given

from myrtlespeech.builders.speech_to_text import build
from myrtlespeech.protos import speech_to_text_pb2
from tests.protos.test_speech_to_text import speech_to_texts


# Tests -----------------------------------------------------------------------


@given(stt_cfg=speech_to_texts())
def test_build_returns_speech_to_text(
    stt_cfg: speech_to_text_pb2.SpeechToText
) -> None:
    """Test that build returns a SpeechToText instance."""
    build(stt_cfg)
    warnings.warn("SpeechToText module only build and not checked if correct")
