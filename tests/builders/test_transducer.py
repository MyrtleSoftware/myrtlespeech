import hypothesis.strategies as st
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.transducer import build as build_transducer

from tests.protos.test_transducer import transducer


# Utilities -------------------------------------------------------------------


@given(
    data=st.data(),
    input_features=st.integers(min_value=2, max_value=16),
    input_channels=st.integers(min_value=1, max_value=5),
    vocab_size=st.integers(min_value=1, max_value=8),
    time_reduction=st.booleans(),
)
@settings(deadline=3000)
def test_build_transducer_does_not_fail(
    data,
    input_features: int,
    input_channels: int,
    vocab_size: int,
    time_reduction: bool,
) -> None:
    """Ensures builder does not throw exception."""
    transducer_cfg = data.draw(transducer(time_reduction=time_reduction))
    build_transducer(
        transducer_cfg, input_features, input_channels, vocab_size
    )
