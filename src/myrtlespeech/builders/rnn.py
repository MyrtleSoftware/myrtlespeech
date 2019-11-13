from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNN
from myrtlespeech.model.rnn import RNNType
from myrtlespeech.protos import rnn_pb2


def build(
    rnn_cfg: rnn_pb2.RNN, input_features: int
) -> Tuple[torch.nn.Module, int]:
    """Returns a :py:class:`myrtlespeech.model.enocder.rnn.RNN` based on cfg.

    Args:
        rnn_cfg: A :py:class:`myrtlespeech.protos.rnn_pb2.RNN` protobuf object
            containing the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

    Returns:
        A tuple containing an :py:class:`myrtlespeech.model.encoder.rnn.RNN`
        based on the config and the number of features that the final layer
        will output.

        The module's :py:meth:`torch.nn.Module.forward` method accepts
        :py:class:`torch.Tensor` input with size ``[max_input_seq_len, batch,
        input_features]`` and produces a :py:class:`torch.Tensor` with size
        ``[max_output_seq_len, batch, output_features]``.

        If the ``seq_lens`` kwarg is passed to
        :py:class:`torch.nn.Module.forward` then the return value will be a
        tuple where the first value is the rnn output and the second value is
        ``seq_lens``.

    Example:
        >>> from google.protobuf import text_format
        >>> rnn_cfg_text = '''
        ... rnn_type: LSTM;
        ... hidden_size: 1024;
        ... num_layers: 5;
        ... bias: true;
        ... bidirectional: true;
        ... '''
        >>> rnn_cfg = text_format.Merge(
        ...     rnn_cfg_text,
        ...     rnn_pb2.RNN()
        ... )
        >>> build(rnn_cfg, input_features=512)
        (Sequential(
          (0): RNN(
            (rnn): LSTM(512, 1024, bidirectional=True)
          )
          (1): RNN(
            (rnn): LSTM(2048, 1024, bidirectional=True)
          )
          (2): RNN(
            (rnn): LSTM(2048, 1024, bidirectional=True)
          )
          (3): RNN(
            (rnn): LSTM(2048, 1024, bidirectional=True)
          )
          (4): RNN(
            (rnn): LSTM(2048, 1024, bidirectional=True)
          )
        ), 2048)

        (RNN(
          (rnn): LSTM(512, 1024, num_layers=5, bidirectional=True)
        ), 2048)
    """
    rnn_type_map = {
        rnn_pb2.RNN.LSTM: RNNType.LSTM,
        rnn_pb2.RNN.GRU: RNNType.GRU,
        rnn_pb2.RNN.BASIC_RNN: RNNType.BASIC_RNN,
    }
    try:
        rnn_type = rnn_type_map[rnn_cfg.rnn_type]
    except KeyError:
        raise ValueError(f"rnn_type={rnn_cfg.rnn_type} not supported")

    forget_gate_bias = None
    if rnn_cfg.HasField("forget_gate_bias"):
        forget_gate_bias = rnn_cfg.forget_gate_bias.value

    rnn_layers = []
    for i in range(rnn_cfg.num_layers):
        # Batch norm is eventually added only after the first layer
        # (if batch_norm == True)
        num_directions = 2 if rnn_cfg.bidirectional else 1
        rnn_layers.append(
            RNN(
                rnn_type=rnn_type,
                input_size=rnn_cfg.hidden_size * num_directions if i > 0
                else input_features,
                hidden_size=rnn_cfg.hidden_size,
                bias=rnn_cfg.bias,
                bidirectional=rnn_cfg.bidirectional,
                forget_gate_bias=forget_gate_bias,
                batch_norm=rnn_cfg.batch_norm if i > 0 else False,
            )
        )
    rnn = torch.nn.Sequential(*rnn_layers)

    out_features = rnn_cfg.hidden_size
    if rnn_cfg.bidirectional:
        out_features *= 2

    return rnn, out_features
