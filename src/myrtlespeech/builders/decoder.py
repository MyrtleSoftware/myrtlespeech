"""
.. todo::

    * add examples in the docstrings for each to make onboarding easier?
"""
import torch

from myrtlespeech.builders.fully_connected import build as build_fully_connected
from myrtlespeech.protos import decoder_pb2


def build(
    decoder_cfg: decoder_pb2.Decoder, input_features: int, output_features: int
) -> torch.nn.Module:
    """Returns an :py:class:`torch.nn.Module` based on the given config.

    Args:
        decoder_cfg: A ``Decoder`` protobuf object containing the config for
            the desired :py:class:`.Decoder`.

        input_features: The number of features for the input.

        output_features: The number of features for the output.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.
    """
    decoder_choice = decoder_cfg.WhichOneof("supported_decoders")
    if decoder_choice == "fully_connected":
        decoder = build_fully_connected(
            decoder_cfg.fully_connected, input_features, output_features
        )
    else:
        raise ValueError(f"{decoder_choice} not supported")

    return decoder
