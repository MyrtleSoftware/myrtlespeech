from hypothesis import given
from myrtlespeech.builders.rnn_t_greedy_decoder import build
from myrtlespeech.post_process.rnn_t_greedy_decoder import RNNTGreedyDecoder
from myrtlespeech.protos import rnn_t_greedy_decoder_pb2

from tests.post_process.test_rnn_t_beam_decoder import DummyRNNTModel
from tests.protos.test_rnn_t_decoders import rnn_t_greedy_decoder

# Utilities -------------------------------------------------------------------


def rnn_t_greedy_decoder_module_match_cfg(
    rnn_t: RNNTGreedyDecoder,
    rnn_t_cfg: rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder,
) -> None:
    """Ensures RNNTGreedyDecoder matches protobuf configuration."""
    assert rnn_t.blank_index == rnn_t_cfg.blank_index
    if rnn_t_cfg.max_symbols_per_step != 0:  # default value
        assert rnn_t.max_symbols_per_step == rnn_t_cfg.max_symbols_per_step
    assert hasattr(rnn_t, "model")


# Tests -----------------------------------------------------------------------


@given(rnn_t_cfg=rnn_t_greedy_decoder())
def test_build_returns_correct_rnn_t_greedy_decoder_with_valid_params(
    rnn_t_cfg: rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder,
) -> None:
    """Test that build returns correct RNNTGreedyDecoder with valid params."""
    rnn_t = DummyRNNTModel()
    rnn_t_decoder = build(rnn_t_cfg, rnn_t)
    rnn_t_greedy_decoder_module_match_cfg(rnn_t_decoder, rnn_t_cfg)
