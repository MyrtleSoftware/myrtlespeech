"""Builds an RNN :py:class:`torch.nn.Module` from a configuration."""
import torch

from myrtlespeech.protos import rnn_pb2


def build_rnn(rnn_cfg: rnn_pb2.RNN, input_features: int) -> torch.nn.Module:
    """Returns a :py:class:`torch.nn.Module` based on the config.

    Args:
        rnn_cfg: A ``RNN`` protobuf object containing the config for the
            desired :py:class:`torch.nn.Module`.

        input_size: The number of features for the input.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.

        The :py:class:`torch.nn.Module` accepts :py:class:`torch.Tensor` input
        of size `[seq_len, batch, input_features]`.
    """
    rnn_type_map = {0: torch.nn.LSTM, 1: torch.nn.GRU, 2: torch.nn.RNN}
    try:
        rnn_type = rnn_type_map[rnn_cfg.rnn_type]
    except KeyError:
        raise ValueError(
            f"build_rnn does not support rnn_type={rnn_cfg.rnn_type}"
        )

    rnn = rnn_type(
        input_size=input_features,
        hidden_size=rnn_cfg.hidden_size,
        num_layers=rnn_cfg.num_layers,
        bias=rnn_cfg.bias,
        bidirectional=rnn_cfg.bidirectional,
    )

    return rnn
