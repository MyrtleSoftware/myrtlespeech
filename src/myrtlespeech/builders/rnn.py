import torch

from myrtlespeech.protos import rnn_pb2
from myrtlespeech.model.encoder.rnn import RNN, RNNType


def build(rnn_cfg: rnn_pb2.RNN, input_features: int) -> torch.nn.Module:
    """Returns a :py:class:`myrtlespeech.model.enocder.rnn.RNN` based on cfg.

    Args:
        rnn_cfg: A :py:class:`myrtlespeech.protos.rnn_pb2.RNN` protobuf object
            containing the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

    Returns:
        A :py:class:`myrtlespeech.model.encoder.rnn.RNN` based on the config.

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
        LSTM(512, 1024, num_layers=5, bidirectional=True)
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

    rnn = RNN(
        rnn_type=rnn_type,
        input_size=input_features,
        hidden_size=rnn_cfg.hidden_size,
        num_layers=rnn_cfg.num_layers,
        bias=rnn_cfg.bias,
        bidirectional=rnn_cfg.bidirectional,
        forget_gate_bias=forget_gate_bias,
    )

    return rnn
