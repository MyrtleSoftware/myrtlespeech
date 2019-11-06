import hypothesis.strategies as st
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.rnn_t import build as build_rnn_t
from myrtlespeech.protos import rnn_t_pb2

from tests.protos.test_rnn_t import rnn_t


# Utilities -------------------------------------------------------------------


@given(
    rnn_t_cfg=rnn_t(),
    input_features=st.integers(min_value=1, max_value=32),
    vocab_size=st.integers(min_value=1, max_value=32),
)
@settings(deadline=3000)
def test_build_rnn_t_does_not_fail(
    rnn_t_cfg: rnn_t_pb2.RNNT, input_features: int, vocab_size: int
) -> None:
    """Ensures builder does not throw exception."""

    actual = build_rnn_t(rnn_t_cfg, input_features, vocab_size)
