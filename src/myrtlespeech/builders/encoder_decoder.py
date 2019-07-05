"""Builds an :py:class:`.EncoderDecoder` model from a configuration.

.. todo::

    * type all the _cfg parameters?

    * add examples in the docstrings for each to make onboarding easier?

    * how to configure input_size for _build_rnn?
"""
from typing import Optional

from myrtlespeech.builders.decoder import build as build_decoder
from myrtlespeech.builders.encoder import build as build_encoder
from myrtlespeech.model.encoder_decoder import EncoderDecoder
from myrtlespeech.protos import encoder_decoder_pb2


def build(
    encoder_decoder_cfg: encoder_decoder_pb2.EncoderDecoder,
    input_features: int,
    output_features: int,
    input_channels: Optional[int],
) -> EncoderDecoder:
    """Returns a :py:class:`.EncoderDecoder` model based on the model config.

    Args:
        encoder_decoder_cfg: A ``EncoderDecoder.proto`` object containing the
            config for the desired :py:class:`.EncoderDecoder`.

        input_features: The number of features for the input.

        input_channels: The number of channels for the input. May be ``None``
            if encoder does require it.

    Returns:
        An :py:class:`.EncoderDecoder` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    if not isinstance(encoder_decoder_cfg, encoder_decoder_pb2.EncoderDecoder):
        raise ValueError(
            "encoder_decoder_cfg not of type encoder_decoder_pb.EncoderDecoder"
        )

    encoder, input_features = build_encoder(
        encoder_decoder_cfg.encoder,
        input_features=input_features,
        input_channels=input_channels,
    )

    decoder = build_decoder(
        encoder_decoder_cfg.decoder,
        input_features=input_features,
        output_features=output_features,
    )

    return EncoderDecoder(encoder=encoder, decoder=decoder)
