"""
.. todo::

    * add examples in the docstrings for each to make onboarding easier?
"""
import torch

from myrtlespeech.model.encoder.vgg import cfgs, make_layers
from myrtlespeech.protos import vgg_pb2


def build(
    vgg_cfg: vgg_pb2.VGG, input_channels: int, seq_len_wrapper: bool = False
) -> torch.nn.Module:
    """Returns a :py:class:`torch.nn.Module` based on the VGG config.

    Args:
        rnn_cfg: A ``RNN`` protobuf object containing the config for the
            desired :py:class:`torch.nn.Module`.

        input_channels: The number of channels -- not features! -- for the
            input.

        seq_len_wrapper: TODO

    Returns:
        A :py:class:`torch.nn.Module` based on the config.

    Example:

        >>> from google.protobuf import text_format
        >>> vgg_cfg_text = '''
        ... vgg_config: A;
        ... batch_norm: false;
        ... use_output_from_block: 2;
        ... '''
        >>> vgg_cfg = text_format.Merge(
        ...     vgg_cfg_text,
        ...     vgg_pb2.VGG()
        ... )
        >>> build(vgg_cfg, input_channels=3)
        Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace)
          (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): ReLU(inplace)
          (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (7): ReLU(inplace)
          (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (9): ReLU(inplace)
          (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
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
        seq_len_wrapper=seq_len_wrapper,
    )

    return vgg
