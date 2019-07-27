from typing import Dict, List, Tuple

import torch


def pad_sequence(
    sequences: List[torch.Tensor], padding_value: int = 0
) -> torch.Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``.

    This is a variant of :py:func:`torch.nn.utils.rnn.pad_sequence` where the
    length is the last dimension in the size instead of the first.

    Args:
        sequences: A list of variable length sequences.
        padding_value: Value for padded elements.

    Returns:
        :py:class:`torch.Tensor` of size ``[batch, *, max_seq_len]``.

    Example:
        >>> a = torch.ones(300, 25)
        >>> b = torch.ones(300, 22)
        >>> c = torch.ones(300, 15)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([3, 300, 25])
    """
    # assume type and size (excluding sequence length) of all Tensors in
    # sequences are the same
    max_size = sequences[0].size()
    leading_dims = max_size[:-1]
    max_len = max([s.size(-1) for s in sequences])

    out_dims = (len(sequences),) + leading_dims + (max_len,)

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        out_tensor[i, ..., :length] = tensor

    return out_tensor


def seq_to_seq_collate_fn(
    batch: List[
        Tuple[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ]
    ]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    r"""Collates a list of ``((tensor, tensor_len), (target, target_len))``.

    A ``collate_fn`` for sequence-to-sequence tasks.

    Args:
        batch: A list of ``((tensor, tensor_len), (target, target_len))`` where:

            tensor:
                A :py:class:`torch.Tensor` of input for a model. The sequence
                length dimension must be last.

            tensor_len:
                A scalar, integer :py:class:`torch.Tensor` giving the length of
                ``tensor``.

            target:
                A :py:class:`torch.Tensor` target for the model. The sequence
                length dimension must be last.

            target_len:
                A scalar, integer :py:class:`torch.Tensor` giving the length of
                ``target``.

    Returns:
        A tuple of two dictionaries.

        The first dictionary has two keys, ``x`` and ``seq_lens``. ``x`` is the
        result of applying :py:func:`.pad_sequence` to all ``tensor``\s and
        ``seq_lens`` is the result of stacking all ``tensor_len``\s.

        The second dictionary has two keys, ``targets`` and ``target_lengths``.
        ``targets`` is the result of appying :py:func:`.pad_sequence` to all
        ``target``\s and ``target_lengths`` is the result of stacking all
        ``target_len``\s.
    """
    inputs, in_seq_lens = [], []
    targets, target_seq_lens = [], []

    for (input, in_seq_len), (target, target_seq_len) in batch:
        inputs.append(input)
        in_seq_lens.append(in_seq_len)
        targets.append(target)
        target_seq_lens.append(target_seq_len)

    inputs = pad_sequence(inputs)
    in_seq_lens = torch.tensor(in_seq_lens, requires_grad=False)
    targets = pad_sequence(targets)
    target_seq_lens = torch.tensor(target_seq_lens, requires_grad=False)

    x = {"x": inputs, "seq_lens": in_seq_lens}
    y = {"targets": targets, "target_lengths": target_seq_lens}

    return x, y
