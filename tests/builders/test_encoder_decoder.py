from typing import Optional

import hypothesis.strategies as st
from hypothesis import given
from myrtlespeech.builders.encoder_decoder import build
from myrtlespeech.protos import encoder_decoder_pb2

from tests.protos.test_encoder_decoder import encoder_decoders


# Tests -----------------------------------------------------------------------


@given(
    encoder_decoder_cfg=encoder_decoders(valid_only=True),
    input_features=st.integers(1, 32),
    output_features=st.integers(1, 32),
    input_channels=st.integers(1, 32),
)
def test_build_returns_encoder_decoder(
    encoder_decoder_cfg: encoder_decoder_pb2.EncoderDecoder,
    input_features: int,
    output_features: int,
    input_channels: int,
) -> None:
    """Test that build returns an EncoderDecoder.

    Does _not_ check whether the returned EncoderDecoder is correct.
    """
    build(encoder_decoder_cfg, input_features, output_features, input_channels)
