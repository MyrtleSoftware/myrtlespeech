from typing import Optional

import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.decoder import build as build_decoder
from myrtlespeech.protos import decoder_pb2
from tests.protos.test_decoder import decoders


# Utilities -------------------------------------------------------------------


# def decoder_module_match_cfg(
#    decoder: Decoder,
#    decoder_cfg: decoder_pb2.Decoder,
#    input_features: int,
#    input_channels: Optional[int],
# ) -> None:
#    """Ensures Decoder matches protobuf configuration."""
#    assert isinstance(decoder, torch.nn.Module)
#
#    # check cnn config
#    if decoder_cfg.HasField("no_cnn"):
#        assert decoder.cnn is None
#    elif decoder_cfg.HasField("vgg"):
#        vgg_module_match_cfg(  # type: ignore
#            decoder.cnn, decoder_cfg.vgg, input_channels
#        )
#        out_size = vgg_output_size(
#            decoder.cnn, torch.Size([-1, input_channels, input_features, -1])
#        )
#        input_features = out_size[1] * out_size[2]
#    else:
#        raise ValueError("expected either no_cnn or vgg to be set")
#
#    # check rnn config
#    if decoder_cfg.HasField("no_rnn"):
#        assert decoder.rnn is None
#    elif decoder_cfg.HasField("rnn"):
#        rnn_module_match_cfg(decoder.rnn, decoder_cfg.rnn, input_features)
#    else:
#        raise ValueError("expected either no_rnn or rnn to be set")


# Tests -----------------------------------------------------------------------


# @given(
#    encoder_cfg=encoders(),
#    input_features=st.integers(min_value=1, max_value=32),
#    input_channels=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
# )
# def test_build_encoder_returns_correct_module_structure(
#    encoder_cfg: encoder_pb2.Encoder,
#    input_features: int,
#    input_channels: Optional[int],
# ) -> None:
#    """Ensures Module returned by ``build_encoder`` has correct structure."""
#    assume(encoder_cfg.HasField("no_cnn") or input_channels is not None)
#    encoder = build_encoder(encoder_cfg, input_features, input_channels)
#    encoder_module_match_cfg(
#        encoder, encoder_cfg, input_features, input_channels
#    )
#
#
# @given(
#    encoder_cfg=encoders(),
#    input_features=st.integers(min_value=1, max_value=32),
# )
# def test_input_channels_none_when_not_no_cnn_raises_value_error(
#    encoder_cfg: encoder_pb2.Encoder, input_features: int
# ) -> None:
#    """Ensures ValueError raised when input_channels is None, cnn not None."""
#    assume(not encoder_cfg.HasField("no_cnn"))
#    with pytest.raises(ValueError):
#        build_encoder(encoder_cfg, input_features, None)
