import hypothesis.strategies as st
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.transducer import build as build_transducer
from myrtlespeech.protos import transducer_pb2

from tests.protos.test_transducer import transducer


# Utilities -------------------------------------------------------------------


@given(
    transducer_cfg=transducer(),
    input_features=st.integers(min_value=2, max_value=32),
    input_channels=st.integers(min_value=1, max_value=5),
    vocab_size=st.integers(min_value=1, max_value=32),
)
@settings(deadline=3000)
def test_build_transducer_does_not_fail(
    transducer_cfg: transducer_pb2.Transducer,
    input_features: int,
    input_channels: int,
    vocab_size: int,
) -> None:
    """Ensures builder does not throw exception."""
    build_transducer(
        transducer_cfg, input_features, input_channels, vocab_size
    )
