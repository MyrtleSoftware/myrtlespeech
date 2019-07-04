"""
.. todo::

    * add examples in the docstrings for each to make onboarding easier?
"""
import torch

from myrtlespeech.model.decoder.fully_connected import FullyConnected
from myrtlespeech.protos import fully_connected_pb2


def build(
    fully_connected_cfg: fully_connected_pb2.FullyConnected,
    input_features: int,
    output_features: int,
) -> FullyConnected:
    """Returns a :py:class:`.FullyConnected` based on the config.

    Args:
        fully_connected_cfg: A ``FullyConnected`` protobuf object containing
            the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

        output_features: The number of output features.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.
    """
    pb2_fc = fully_connected_pb2.FullyConnected
    if fully_connected_cfg.hidden_activation_fn == pb2_fc.RELU:
        hidden_activation_fn = torch.nn.ReLU()
    elif fully_connected_cfg.hidden_activation_fn == pb2_fc.NONE:
        hidden_activation_fn = None
    else:
        raise ValueError("unsupported activation_fn")

    hidden_size = None
    if fully_connected_cfg.hidden_size > 0:
        hidden_size = fully_connected_cfg.hidden_size

    return FullyConnected(
        in_features=input_features,
        out_features=output_features,
        num_hidden_layers=fully_connected_cfg.num_hidden_layers,
        hidden_size=hidden_size,
        hidden_activation_fn=hidden_activation_fn,
    )
