import torch
from myrtlespeech.protos import activation_pb2


def build(activation_cfg: activation_pb2.Activation) -> torch.nn.Module:
    """Returns a activation function based on the config.

    Args:
        activation_cfg: A ``Activation`` protobuf object containing the config
            for the desired :py:class:`torch.nn.Module`.

    Returns:
        An activation function (:py:class:`torch.nn.Module`) based on the
        config.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... activation {
        ...   hardtanh {
        ...     min_val: 0.0;
        ...     max_val: 20.0;
        ...   }
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     activation_pb2.Activation()
        ... )
        >>> build(cfg)
        Hardtanh(min_val=0.0, max_val=20.0)
    """
    if activation_cfg.HasField("identity"):
        return torch.nn.Identity()
    elif activation_cfg.HasField("hardtanh"):
        return torch.nn.Hardtanh(
            min_val=activation_cfg.hardtanh.min_val,
            max_val=activation_cfg.hardtanh.max_val,
        )
    elif activation_cfg.HasField("relu"):
        return torch.nn.ReLU()
    else:
        raise ValueError(f"unsupported activation_cfg {activation_cfg}")
