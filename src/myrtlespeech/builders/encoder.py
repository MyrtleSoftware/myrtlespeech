"""
.. todo::

    * add examples in the docstrings for each to make onboarding easier?
"""
from typing import Tuple, Optional

import torch

from myrtlespeech.builders.rnn import build_rnn
from myrtlespeech.builders.vgg import build_vgg
from myrtlespeech.model.encoder.encoder import Encoder
from myrtlespeech.model.encoder.vgg import vgg_output_size
from myrtlespeech.protos import encoder_pb2


def build_encoder(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: Optional[int] = None,
) -> Tuple[Encoder, torch.Size]:
    """Returns an :py:class:`.Encoder` based on the given config.

    Args:
        encoder_cfg: A ``Encoder`` protobuf object containing the config for
            the desired :py:class:`.Encoder`.

        input_features: The number of features for the input.

        input_channels: The number of channels for the input. May be ``None``
            if ``encoder_cfg`` does not use a ``cnn``.

    Returns:
        A tuple containing an :py:class:`.Encoder` based on the config.
    """
    # build cnn, if any
    cnn_choice = encoder_cfg.WhichOneof("supported_cnns")
    if cnn_choice != "no_cnn" and input_channels is None:
        raise ValueError("input_channels must not be None when cnn set")
    if cnn_choice == "no_cnn":
        cnn = None
    elif cnn_choice == "vgg":
        cnn = build_vgg(encoder_cfg.vgg, input_channels)  # type: ignore
        # update number of features based on vgg output
        out_size = vgg_output_size(
            cnn, torch.Size([-1, input_channels, input_features, -1])
        )
        input_features = out_size[1] * out_size[2]
    else:
        raise ValueError(f"{cnn_choice} not supported")

    # build rnn, if any
    rnn_choice = encoder_cfg.WhichOneof("supported_rnns")
    if rnn_choice == "no_rnn":
        rnn = None
    elif rnn_choice == "rnn":
        rnn = build_rnn(encoder_cfg.rnn, input_features)
    else:
        raise ValueError(f"{rnn_choice} not supported")

    return Encoder(cnn=cnn, rnn=rnn)
