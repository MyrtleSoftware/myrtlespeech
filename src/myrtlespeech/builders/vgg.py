"""
.. todo::

    * add examples in the docstrings for each to make onboarding easier?
"""
import torch

from myrtlespeech.model.encoder.vgg import cfgs, make_layers
from myrtlespeech.protos import vgg_pb2


def build_vgg(vgg_cfg: vgg_pb2.VGG, input_channels: int) -> torch.nn.Module:
    """Returns a :py:class:`torch.nn.Module` based on the VGG config.

    Args:
        rnn_cfg: A ``RNN`` protobuf object containing the config for the
            desired :py:class:`torch.nn.Module`.

        input_channels: The number of channels -- not features! -- for the
            input.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.
    """
    vgg_config_map = {
        index: letter for letter, index in vgg_pb2.VGG.VGG_CONFIG.items()
    }
    try:
        vgg_config = vgg_config_map[vgg_cfg.vgg_config]
    except KeyError:
        raise ValueError(f"vgg_config={vgg_cfg.vgg_config} not supported")

    vgg = make_layers(
        cfg=cfgs[vgg_config],
        in_channels=input_channels,
        batch_norm=vgg_cfg.batch_norm,
        use_output_from_block=vgg_cfg.use_output_from_block + 1,
    )

    return vgg
