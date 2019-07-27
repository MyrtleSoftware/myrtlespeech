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


def collate_fn(
    batch: List[
        Tuple[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ]
    ]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """TODO: refactor/document/test/move to appropriate place"""
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

    xb = {"x": inputs, "seq_lens": in_seq_lens}
    yb = {"targets": targets, "target_lengths": target_seq_lens}

    return xb, yb
