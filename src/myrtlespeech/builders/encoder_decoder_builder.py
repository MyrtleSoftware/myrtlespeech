"""Builds an :py:class:`.EncoderDecoder` model from a configuration.

.. todo::

    * type all the _cfg parameters?

    * add examples in the docstrings for each to make onboarding easier?

    * how to configure input_size for _build_rnn?
"""
import torch

from myrtlespeech.model.encoder_decoder import EncoderDecoder
from myrtlespeech.model.encoder.encoder import Encoder
from myrtlespeech.model.encoder.vgg import cfgs, make_layers
from myrtlespeech.protos import encoder_decoder_pb2


def build(encoder_decoder_cfg) -> EncoderDecoder:
    """Returns a :py:class:`.EncoderDecoder` model based on the model config.

    Args:
        encoder_decoder_cfg: A ``EncoderDecoder.proto`` object containing the
            config for the desired :py:class:`.EncoderDecoder`.

    Returns:
        An :py:class:`.EncoderDecoder` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    if not isinstance(encoder_decoder_cfg, encoder_decoder_pb2.EncoderDecoder):
        raise ValueError(
            "encoder_decoder_cfg not of type encoder_decoder_pb.EncoderDecoder"
        )

    encoder = _build_encoder(encoder_decoder_cfg.encoder)

    decoder = None

    return EncoderDecoder(encoder=encoder, decoder=decoder)


def _build_encoder(encoder_cfg) -> Encoder:
    """Returns a :py:class:`.Encoder` based on the given config.

    Args:
        rnn_encoder_cfg: A ``Encoder.proto`` object containing the config for
            the desired :py:class:`.Encoder`.

    Returns:
        A :py:class:`.Encoder` based on the config.
    """
    # build cnn, if any
    cnn_choice = encoder_cfg.WhichOneof("supported_cnns")
    if cnn_choice == "no_cnn":
        cnn = None
    elif cnn_choice == "vgg":
        cnn = _build_vgg(encoder_cfg.vgg)
    else:
        raise ValueError(f"_build_encoder does not support {cnn_choice}")

    # build rnn, if any
    rnn_choice = encoder_cfg.WhichOneof("supported_rnns")
    if rnn_choice == "no_rnn":
        rnn = None
    elif rnn_choice == "rnn":
        rnn = _build_rnn(encoder_cfg.rnn)
    else:
        raise ValueError(f"_build_encoder does not support {rnn_choice}")

    return Encoder(cnn=cnn, rnn=rnn)


def _build_vgg(vgg_cfg) -> torch.nn.Module:
    """Returns a :py:class:`torch.nn.Module` based on the VGG confg.

    Args:
        vgg_cfg: A ``VGG.proto`` object containing the config for the desired
            :py:class:`torch.nn.Module`.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.
    """
    vgg_config_map = {0: "A", 1: "B", 3: "D", 4: "E"}
    try:
        vgg_config = vgg_config_map[vgg_cfg.vgg_config]
    except KeyError:
        raise ValueError(
            f"_build_vgg does not support vgg_config={vgg_cfg.vgg_config}"
        )

    vgg = make_layers(
        cfg=cfgs[vgg_config],
        batch_norm=vgg_cfg.batch_norm,
        use_output_from_block=vgg_cfg.use_output_from_block + 1,
    )

    return vgg


def _build_rnn(rnn_cfg) -> torch.nn.Module:
    """Returns a :py:class:`torch.nn.Module` based on the config.

    .. todo:: what is the input size!?

    Args:
        rnn_cfg: A ``RNN.proto`` object containing the config for the desired
            :py:class:`torch.nn.Module`.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.
    """
    rnn_type_map = {0: torch.nn.LSTM, 1: torch.nn.GRU, 2: torch.nn.RNN}
    try:
        rnn_type = rnn_type_map[rnn_cfg.rnn_type]
    except KeyError:
        raise ValueError(f"_build_rnn does not rnn_type={rnn_cfg.rnn_type}")

    return rnn_type(
        input_size=1,  # TODO
        hidden_size=rnn_cfg.hidden_size,
        num_layers=rnn_cfg.num_layers,
        bias=rnn_cfg.bias,
        bidirectional=rnn_cfg.bidirectional,
    )
