from hypothesis import given
from myrtlespeech.builders.transducer_greedy_decoder import build
from myrtlespeech.post_process.transducer_greedy_decoder import (
    TransducerGreedyDecoder,
)
from myrtlespeech.protos import transducer_greedy_decoder_pb2

from tests.post_process.test_transducer_beam_decoder import (
    get_dummy_transducer,
)
from tests.protos.test_transducer_decoders import transducer_greedy_decoder

# Utilities -------------------------------------------------------------------


def transducer_greedy_module_match_cfg(
    transducer: TransducerGreedyDecoder,
    transducer_cfg: transducer_greedy_decoder_pb2.TransducerGreedyDecoder,
) -> None:
    """Ensures TransducerGreedyDecoder matches protobuf configuration."""
    assert transducer.blank_index == transducer_cfg.blank_index
    if transducer_cfg.max_symbols_per_step != 0:  # default value
        assert (
            transducer.max_symbols_per_step
            == transducer_cfg.max_symbols_per_step
        )
    assert hasattr(transducer, "model")


# Tests -----------------------------------------------------------------------


@given(transducer_cfg=transducer_greedy_decoder())
def test_build_returns_correct_transducer_greedy_decoder_with_valid_params(
    transducer_cfg: transducer_greedy_decoder_pb2.TransducerGreedyDecoder,
) -> None:
    """Test build returns correct TransducerGreedyDecoder with valid params."""
    transducer = get_dummy_transducer()
    transducer_decoder = build(transducer_cfg, transducer)
    transducer_greedy_module_match_cfg(transducer_decoder, transducer_cfg)
