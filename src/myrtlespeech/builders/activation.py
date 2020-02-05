import torch
from myrtlespeech.protos import activation_pb2


def build(activation_cfg: activation_pb2.Activation) -> torch.nn.Module:
    """Returns an activation function based on the config.

    Args:
        activation_cfg: An ``Activation`` protobuf object containing the config
            for the desired activation function (:py:class:`torch.nn.Module`).

    Returns:
        An activation function (:py:class:`torch.nn.Module`) based on the
        config.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... hardtanh {
        ...   min_val: 0.0;
        ...   max_val: 20.0;
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     activation_pb2.Activation()
        ... )
        >>> build(cfg)
        Hardtanh(min_val=0.0, max_val=20.0)
    """
    act_str = activation_cfg.WhichOneof("activation")
    if act_str == "identity":
        return torch.nn.Identity()
    elif act_str == "hardtanh":
        return torch.nn.Hardtanh(
            min_val=float(activation_cfg.hardtanh.min_val),
            max_val=float(activation_cfg.hardtanh.max_val),
        )
    elif act_str == "relu":
        return torch.nn.ReLU()
    else:
        raise ValueError(f"unsupported activation_cfg {activation_cfg}")
