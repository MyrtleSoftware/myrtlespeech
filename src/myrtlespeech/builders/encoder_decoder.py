"""Builds an :py:class:`.EncoderDecoder` model from a configuration.

.. todo::

    * type all the _cfg parameters?

    * add examples in the docstrings for each to make onboarding easier?

    * how to configure input_size for _build_rnn?
"""
import torch

from myrtlespeech.builders.encoder import build as build_encoder
from myrtlespeech.model.encoder_decoder import EncoderDecoder
from myrtlespeech.protos import encoder_decoder_pb2


def build(encoder_decoder_cfg, input_size: torch.Size) -> EncoderDecoder:
    """Returns a :py:class:`.EncoderDecoder` model based on the model config.

    Args:
        encoder_decoder_cfg: A ``EncoderDecoder.proto`` object containing the
            config for the desired :py:class:`.EncoderDecoder`.

        input_size: A :py:class:`torch.Size` object containing the size of the
            input. ``-1`` represents an unknown/dynamic size.

    Returns:
        An :py:class:`.EncoderDecoder` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    if not isinstance(encoder_decoder_cfg, encoder_decoder_pb2.EncoderDecoder):
        raise ValueError(
            "encoder_decoder_cfg not of type encoder_decoder_pb.EncoderDecoder"
        )

    encoder = build_encoder(encoder_decoder_cfg.encoder, input_size)

    decoder = None

    return EncoderDecoder(encoder=encoder, decoder=decoder)
