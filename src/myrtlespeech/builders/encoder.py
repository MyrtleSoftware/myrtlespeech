"""Builds an :py:class:`.EncoderDecoder` model from a configuration.

.. todo::

    * type all the _cfg parameters?

    * add examples in the docstrings for each to make onboarding easier?

    * how to configure input_size for _build_rnn?
"""
from typing import Tuple

import torch

from myrtlespeech.model.encoder.encoder import Encoder
from myrtlespeech.builders.vgg import build_vgg
from myrtlespeech.builders.rnn import build_rnn


def build_encoder(
    encoder_cfg, input_size: torch.Size
) -> Tuple[Encoder, torch.Size]:
    """Returns a :py:class:`.Encoder` based on the given config.

    Args:
        rnn_encoder_cfg: A ``Encoder.proto`` object containing the config for
            the desired :py:class:`.Encoder`.

        input_size: A :py:class:`torch.Size` object containing the size of the
            input. ``-1`` represents an unknown/dynamic size.

    Returns:
        A tuple containing an :py:class:`.Encoder` based on the config and the
        output :py:class:`torch.Size` after :py:class:`.Encoder` is applied to
        input of ``input_size``.
    """
    # build cnn, if any
    cnn_choice = encoder_cfg.WhichOneof("supported_cnns")
    if cnn_choice == "no_cnn":
        cnn = None
    elif cnn_choice == "vgg":
        cnn, input_size = build_vgg(encoder_cfg.vgg, input_size)
    else:
        raise ValueError(f"build_encoder does not support {cnn_choice}")

    # build rnn, if any
    rnn_choice = encoder_cfg.WhichOneof("supported_rnns")
    if rnn_choice == "no_rnn":
        rnn = None
    elif rnn_choice == "rnn":
        rnn, input_size = build_rnn(encoder_cfg.rnn, input_size)
    else:
        raise ValueError(f"build_encoder does not support {rnn_choice}")

    return Encoder(cnn=cnn, rnn=rnn), input_size
