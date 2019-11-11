from typing import List
from typing import Tuple
from typing import Union

import numpy as np
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
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]
    ]
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]:
    r"""Collates a list of ``((tensor, tensor_len), (target, target_len))``.

    A ``collate_fn`` for sequence-to-sequence tasks.

    Args:
        batch: A list of ``((tensor, tensor_len), (target, target_len))``
            where:

            tensor:
                A :py:class:`torch.Tensor` of input for a model. The sequence
                length dimension must be last.

            tensor_len:collate_label_list
                A scalar, integer :py:class:`torch.Tensor` giving the length of
                ``tensor``.

            target:
                A :py:class:`torch.Tensor` target for the model. The sequence
                length dimension must be last.

            target_len:
                A scalar, integer :py:class:`torch.Tensor` giving the length of
                ``target``.

    Returns:
        A tuple of ``((batch_tensor, batch_tensor_len), (batch_target,
        batch_target_len))`` where ``batch_tensor`` is the
        result of applying :py:func:`.pad_sequence` to all ``tensor``\s,
        ``batch_tensor_lens`` is the result of stacking all ``tensor_len``\s,
        ``batch_target`` is the result of appying :py:func:`.pad_sequence` to
        all ``target``\s and ``batch_target_len`` is the result of stacking all
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

    return (inputs, in_seq_lens), (targets, target_seq_lens)


def collate_label_list(
    labels: List[List[int]], device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates a list of label inputs from a list of label indexes.

    Returns a single fully padded `torch.tensor` and a `torch.tensor` of
    lenghts. Created for use in rnnt decoders.

    Args:
        labels: List of length ``[batch]`` where each element is a List of
            ints (the indexes of symbols in the stt alphabet).
        device: string representing a torch.device

    Returns:
        A tuple where the first element is the target label tensor of
            size ``[batch, max_label_length]`` and the second is a
            :py:class:`torch.Tensor` of size ``[batch]`` that contains the
            lengths of these target label sequences.
    """

    if (
        isinstance(labels, list)
        and isinstance(labels[0], list)
        and isinstance(labels[0][0], int)
    ):

        batch_size = len(labels)

        fill_token = -2  # i.e. choose a symbol that is *not* valid
        max_len = max(len(l) for l in labels)

        lengths = torch.IntTensor([len(l) for l in labels]).to(device)

        collated_labels = np.full(
            (batch_size, max_len), fill_value=fill_token, dtype=np.int32
        )

        for e, l in enumerate(labels):
            collated_labels[e, : len(l)] = torch.IntTensor(l)

        return (torch.IntTensor(collated_labels).to(device), lengths)

    raise ValueError(f"`labels` should be of type `List[List[int]]`")
