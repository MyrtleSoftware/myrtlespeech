import torch
from myrtlespeech.builders.activation import build as build_activation
from myrtlespeech.model.fully_connected import FullyConnected
from myrtlespeech.protos import fully_connected_pb2


def build(
        fully_connected_cfg: fully_connected_pb2.FullyConnected,
        input_features: int,
        output_features: int,
) -> torch.nn.Module:
    """Returns a sequence of :py:class:`.FullyConnected` layers based on the
    config, grouped in a :py:class:`torch.nn.Sequential` module.

    Args:
        fully_connected_cfg: A ``FullyConnected`` protobuf object containing
            the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

        output_features: The number of output features.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.

    Raises:
        :py:class:`ValueError`: If ``num_hidden_layers < 0``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and hidden_size is
        not None``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and
        hidden_activation_fn is not None``.

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

    num_hidden_layers = fully_connected_cfg.num_hidden_layers
    hidden_size = None
    if fully_connected_cfg.hidden_size > 0:
        hidden_size = fully_connected_cfg.hidden_size

    if num_hidden_layers < 0:
        raise ValueError("num_hidden_layers must be >= 0")

    if num_hidden_layers == 0:
        if hidden_size is not None:
            raise ValueError(
                "num_hidden_layers==0 but hidden_size is not None"
            )
        if activation is not None:
            raise ValueError(
                "num_hidden_layers==0 but hidden_activation_fn is not None"
            )

    hidden_layers = []
    for i in range(num_hidden_layers + 1):
        # Hidden activation is eventually added only to the hidden layers before
        # the last FullyConnected layer. The same is for the batch norm layers.
        hidden_layers.append(FullyConnected(
            in_features=input_features,
            out_features=hidden_size if i < num_hidden_layers
            else output_features,
            hidden_activation_fn=activation if i < num_hidden_layers else None,
            batch_norm=fully_connected_cfg.batch_norm if i < num_hidden_layers
            else False,
        ))
        assert hidden_size is not None
        input_features = hidden_size

    module = torch.nn.Sequential(*hidden_layers)

    return module
