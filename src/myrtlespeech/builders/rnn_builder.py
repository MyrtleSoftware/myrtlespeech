"""Builds an RNN :py:class:`torch.nn.Module` from a configuration."""
from typing import Tuple

import torch

from myrtlespeech.protos import rnn_pb2


def build_rnn(
    rnn_cfg: rnn_pb2.RNN, input_size: torch.Size
) -> Tuple[torch.nn.Module, torch.Size]:
    """Returns a :py:class:`torch.nn.Module` based on the config.

    Args:
        rnn_cfg: A ``RNN`` protobuf object containing the config for the
            desired :py:class:`torch.nn.Module`.

        input_size: A :py:class:`torch.Size` object representing the size of
            the input. ``-1`` denotes an unknown or dynamic size.

            This must be 3-dimensional where each dimension represents:
                1. Batch size. Likely to be ``-1`` (unknown).
                2. Sequence length. Likely to be ``-1`` (dynamic).
                3. Number of features.

    Returns:
        A tuple. The first element is a :py:class:`torch.nn.Module` based on
        the config. The second element is a 3-dimensional output
        :py:class:`torch.Size` representing the output size after the
        :py:class:`torch.nn.Module` is applied to input of ``input_size``. The
        output :py:class:`torch.Size` will have the same batch size as
        ``input_size`` but the sequence length and number of features may
        change depending on the exact ``rnn_cfg``.
    """
    if len(input_size) != 3:
        raise ValueError("rnn input_size should be 3-dimensional")

    rnn_type_map = {0: torch.nn.LSTM, 1: torch.nn.GRU, 2: torch.nn.RNN}
    try:
        rnn_type = rnn_type_map[rnn_cfg.rnn_type]
    except KeyError:
        raise ValueError(
            f"build_rnn does not support rnn_type={rnn_cfg.rnn_type}"
        )

    rnn = rnn_type(
        input_size=input_size[2],
        hidden_size=rnn_cfg.hidden_size,
        num_layers=rnn_cfg.num_layers,
        bias=rnn_cfg.bias,
        batch_first=True,
        bidirectional=rnn_cfg.bidirectional,
    )

    # update size
    out_features = rnn_cfg.hidden_size
    if rnn_cfg.bidirectional:
        out_features *= 2
    output_size = torch.Size([input_size[0], input_size[1], out_features])

    return rnn, output_size
