from hypothesis import given

from myrtlespeech.builders.encoder_decoder import build
from tests.protos.test_encoder_decoder import encoder_decoders


# Tests -----------------------------------------------------------------------


@given(encoder_decoder_cfg=encoder_decoders())
def test_build_returns_encoder_decoder(encoder_decoder_cfg):
    """Test that build returns an EncoderDecoder.

    Does _not_ check whether the returned EncoderDecoder is correct.
    """
    build(encoder_decoder_cfg)
