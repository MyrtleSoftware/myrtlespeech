"""Builds an RNN :py:class:`torch.nn.Module` from a configuration."""
import torch

from myrtlespeech.model.seq_len_wrapper import SeqLenWrapper
from myrtlespeech.protos import rnn_pb2


def build(
    rnn_cfg: rnn_pb2.RNN, input_features: int, seq_len_support: bool = False
) -> torch.nn.Module:
    """Returns a :py:class:`torch.nn.Module` based on the config.

    Args:
        rnn_cfg: A :py:class:`myrtlespeech.protos.rnn_pb2.RNN` protobuf object
            containing the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

        seq_len_support: If :py:data:`True`, the returned module's
            :py:meth:`torch.nn.Module.forward` method optionally accepts a
            ``seq_lens`` kwarg. The value of this argument must either be
            :py:data:`None` (the default) or be a :py:class:`torch.Tensor` of
            size ``[batch]`` where each entry is an integer that gives the
            sequence length of the corresponding *input* sequence.

            When the ``seq_lens`` argument is not :py:data:`None` the module
            will return a tuple of ``(output, output_seq_lens)``. Here
            ``output`` is the result of applying the module to the input
            sequence and ``output_seq_lens`` is a :py:class:`torch.Tensor` of
            size ``[batch]`` where each entry is an integer that gives the
            sequence length of the corresponding *output* sequence.

    Returns:
        A :py:class:`torch.nn.Module` based on the config. This will be a
        :py:class:`torch.nn.RNN`, :py:class:`torch.nn.LSTM` or
        :py:class:`torch.nn.GRU` module that is wrapped in a
        :py:class:`.SeqLenWrapper` if ``seq_len_support`` is :py:data:`True`.

        The module's :py:meth:`torch.nn.Module.forward` method accepts
        :py:class:`torch.Tensor` input with size ``[max_input_seq_len, batch,
        input_features]`` and produces a :py:class:`torch.Tensor` with size
        ``[max_output_seq_len, batch, output_features]``.

        If ``seq_len_support=True`` and the ``seq_lens`` argument is passed to
        :py:class:`torch.nn.Module.forward` then the return value will be a
        tuple as described in the ``seq_len_support`` part of the Args section.

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
    rnn_type_map = {0: torch.nn.LSTM, 1: torch.nn.GRU, 2: torch.nn.RNN}
    try:
        rnn_type = rnn_type_map[rnn_cfg.rnn_type]
    except KeyError:
        raise ValueError(f"rnn_type={rnn_cfg.rnn_type} not supported")

    rnn = rnn_type(
        input_size=input_features,
        hidden_size=rnn_cfg.hidden_size,
        num_layers=rnn_cfg.num_layers,
        bias=rnn_cfg.bias,
        bidirectional=rnn_cfg.bidirectional,
    )

    if not seq_len_support:
        return rnn

    return SeqLenWrapper(module=rnn, seq_lens_fn=lambda seq_lens: seq_lens)
