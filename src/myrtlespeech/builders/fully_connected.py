import torch
from myrtlespeech.builders.activation import build as build_activation
from myrtlespeech.model.fully_connected import FullyConnected
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

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... num_hidden_layers: 2;
        ... hidden_size: 64;
        ... activation {
        ...   relu { }
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     fully_connected_pb2.FullyConnected()
        ... )
        >>> build(cfg, input_features=32, output_features=16)
        FullyConnected(
          (fully_connected): Sequential(
            (0): Linear(in_features=32, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=16, bias=True)
          )
        )
    """
    activation = build_activation(fully_connected_cfg.activation)
    if isinstance(activation, torch.nn.Identity):
        activation = None

    hidden_size = None
    if fully_connected_cfg.hidden_size > 0:
        hidden_size = fully_connected_cfg.hidden_size

    dropout = 0
    if fully_connected_cfg.dropout > 1e-8:
        dropout = fully_connected_cfg.dropout

    return FullyConnected(
        in_features=input_features,
        out_features=output_features,
        num_hidden_layers=fully_connected_cfg.num_hidden_layers,
        hidden_size=hidden_size,
        hidden_activation_fn=activation,
        dropout=dropout,
    )
