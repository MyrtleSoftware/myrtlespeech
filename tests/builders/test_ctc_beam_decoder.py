import hypothesis.strategies as st
import pytest
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.ctc_beam_decoder import build
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.protos import ctc_beam_decoder_pb2

from tests.builders.test_language_model import language_model_module_match_cfg
from tests.protos.test_ctc_beam_decoder import ctc_beam_decoders


# Utilities -------------------------------------------------------------------


def ctc_beam_decoder_module_match_cfg(
    ctc: CTCBeamDecoder, ctc_cfg: ctc_beam_decoder_pb2.CTCBeamDecoder
) -> None:
    """Ensures CTCBeamDecoder matches protobuf configuration."""
    assert ctc.blank_index == ctc_cfg.blank_index
    assert ctc.beam_width == ctc_cfg.beam_width
    assert ctc.prune_threshold == ctc_cfg.prune_threshold

    language_model_module_match_cfg(ctc.language_model, ctc_cfg.language_model)

    if ctc_cfg.HasField("lm_weight"):
        assert ctc.lm_weight == ctc_cfg.lm_weight.value
    else:
        assert ctc.lm_weight is None

    if ctc_cfg.HasField("separator_index"):
        assert ctc.separator_index == ctc_cfg.separator_index.value
    else:
        assert ctc.separator_index is None

    assert ctc.word_weight == ctc_cfg.word_weight


# Tests -----------------------------------------------------------------------


@given(ctc_cfg=ctc_beam_decoders())
def test_build_returns_correct_ctc_beam_decoder_with_valid_params(
    ctc_cfg: ctc_beam_decoder_pb2.CTCBeamDecoder,
) -> None:
    """Test that build returns the correct CTCBeamDecoder with valid params."""
    ctc = build(ctc_cfg)
    ctc_beam_decoder_module_match_cfg(ctc, ctc_cfg)
