from hypothesis import given
from myrtlespeech.builders.transducer_beam_decoder import build
from myrtlespeech.post_process.transducer_beam_decoder import (
    TransducerBeamDecoder,
)
from myrtlespeech.protos import transducer_beam_decoder_pb2

from tests.post_process.test_transducer_beam_decoder import (
    get_dummy_transducer,
)
from tests.protos.test_transducer_decoders import transducer_beam_decoder

# Utilities -------------------------------------------------------------------


def decoder_module_match_cfg(
    transducer: TransducerBeamDecoder,
    transducer_cfg: transducer_beam_decoder_pb2.TransducerBeamDecoder,
) -> None:
    """Ensures TransducerBeamDecoder matches protobuf configuration."""
    assert transducer._blank_index == transducer_cfg.blank_index
    assert transducer._beam_width == transducer_cfg.beam_width
    assert transducer._length_norm == transducer_cfg.length_norm
    if transducer_cfg.max_symbols_per_step != 0:  # default value
        assert (
            transducer._max_symbols_per_step
            == transducer_cfg.max_symbols_per_step
        )
    assert hasattr(transducer, "_log_prune_threshold")
    assert hasattr(transducer, "_model")


# Tests -----------------------------------------------------------------------


@given(transducer_cfg=transducer_beam_decoder())
def test_build_returns_correct_transducer_beam_decoder_with_valid_params(
    transducer_cfg: transducer_beam_decoder_pb2.TransducerBeamDecoder,
) -> None:
    """Test build returns correct TransducerBeamDecoder with valid params."""
    transducer = get_dummy_transducer()
    transducer_decoder = build(transducer_cfg, transducer)
    decoder_module_match_cfg(transducer_decoder, transducer_cfg)
