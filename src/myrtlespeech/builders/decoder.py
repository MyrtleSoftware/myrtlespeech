import torch
from myrtlespeech.builders.fully_connected import build as build_fully_connected
from myrtlespeech.protos import decoder_pb2


def build(
    decoder_cfg: decoder_pb2.Decoder,
    input_features: int,
    output_features: int,
    seq_len_support: bool = False,
) -> torch.nn.Module:
    """Returns an :py:class:`torch.nn.Module` based on the given config.

    Args:
        decoder_cfg: A :py:class:`myrtlespeech.protos.decoder_pb2.Decoder`
            protobuf object containing the config for the desired decoder.

        input_features: The number of features for the input.

        output_features: The number of features for the output.

        seq_len_support: If :py:data:`True`, the returned decoder's
            :py:meth:`torch.nn.Module.forward` method optionally accepts a
            ``seq_lens`` kwarg. The value of this argument must either be
            :py:data:`None` (the default) or be a :py:class:`torch.Tensor` of
            size ``[batch]`` where each entry is an integer that gives the
            sequence length of the corresponding *input* sequence.

            When the ``seq_lens`` argument is not :py:data:`None` the decoder
            will return a tuple of ``(output, output_seq_lens)``. Here
            ``output`` is the result of applying the decoder to the input
            sequence and ``output_seq_lens`` is a :py:class:`torch.Tensor` of
            size ``[batch]`` where each entry is an integer that gives the
            sequence length of the corresponding *output* sequence.

            .. note::
                The :py:meth:`torch.nn.Module.forward` method of a decoder when
                ``seq_len_support=False`` may also support the ``seq_lens``
                argument. This behaviour is not guarenteed and should not be
                relied upon.

    Returns:
        A :py:class:`torch.nn.Module` based on the config.

        The module's :py:meth:`torch.nn.Module.forward` method accepts
        :py:class:`torch.Tensor` input with size ``[max_input_seq_len, batch,
        input_features]`` and produces a :py:class:`torch.Tensor` with size
        ``[max_output_seq_len, batch, output_features]``.

        If ``seq_len_support=True`` and the ``seq_lens`` argument is passed to
        :py:class:`torch.nn.Module.forward` then the return value will be a
        tuple as described in the ``seq_len_support`` part of the Args section.

    Example:
        >>> from google.protobuf import text_format
        >>> decoder_cfg = text_format.Merge('''
        ... fully_connected {
        ...   num_hidden_layers: 0;
        ... }
        ... ''', decoder_pb2.Decoder())
        >>> decoder = build(
        ...     decoder_cfg,
        ...     input_features=10,
        ...     output_features=20,
        ...     seq_len_support=True
        ... )
        >>> # create and process example tensor with shape
        >>> # [max_seq_len, batch, in_features]
        >>> ex_in = torch.empty([100, 3, 10]).normal_()
        >>> # fully connected layers do not change the sequence length
        >>> ex_out = decoder(ex_in)
        >>> ex_out.size()
        torch.Size([100, 3, 20])
        >>> # hence the seq_lens argument also does not change
        >>> ex_in_seq_lens = torch.tensor([10, 25, 100])
        >>> ex_out, ex_out_seq_lens = decoder(ex_in, seq_lens=ex_in_seq_lens)
        >>> ex_out.size()
        torch.Size([100, 3, 20])
        >>> bool(torch.all(ex_out_seq_lens == ex_in_seq_lens))
        True
    """
    decoder_choice = decoder_cfg.WhichOneof("supported_decoders")
    if decoder_choice == "fully_connected":
        decoder = build_fully_connected(
            decoder_cfg.fully_connected, input_features, output_features
        )
    else:
        raise ValueError(f"{decoder_choice} not supported")

    return decoder
