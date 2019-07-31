import random
from typing import List

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from myrtlespeech.data.batch import pad_sequence
from myrtlespeech.data.batch import seq_to_seq_collate_fn
from torch import rand
from torch import tensor

from tests.utils.utils import tensors
from tests.utils.utils import torch_np_dtypes


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def tensor_sequences(draw) -> st.SearchStrategy[List[torch.Tensor]]:
    """Returns a search strategy for lists of Tensors with different seq len.

    The Tensors will have the same dtype and all dimensions except the last
    are guaranteed to be the same.
    """
    dtype = draw(torch_np_dtypes())

    # create at least one tensor, all other tensors in sequences will have same
    # dtype and size apart from the seq length
    base_tensor = draw(tensors(min_n_dims=1, dtype=dtype))
    max_seq_len = base_tensor.size(-1)

    sequences = [base_tensor]
    for _ in range(draw(st.integers(min_value=1, max_value=32))):
        # update size to have different sequence length
        size = list(base_tensor.size())
        size[-1] = draw(st.integers(1, max_seq_len))

        # create new tensor and append to sequences
        seq = base_tensor.new().resize_(size)
        # fill with random values
        if dtype == np.float16:
            seq = seq.to(torch.float32).random_().to(torch.float16)
        else:
            seq.random_()

        sequences.append(seq)

    # shuffle to ensure tensor with maximum sequence length is not always first
    random.shuffle(sequences)

    return sequences


# Tests -----------------------------------------------------------------------

# pad_sequence --------------------------------------


@given(sequences=tensor_sequences())
def test_pad_sequences_returns_tensor_with_correct_size(
    sequences: List[torch.Tensor]
) -> None:
    """Ensures Tensor returned by pad_sequences has correct size."""
    # find longest tensor
    exp_size = max(sequences, key=lambda seq: seq.size(-1)).size()

    # add in batch dimension
    exp_size = torch.Size([len(sequences)] + list(exp_size))

    # compute actual output
    out = pad_sequence(sequences)

    assert out.size() == exp_size


@given(
    sequences=tensor_sequences(),
    padding_value=st.integers(min_value=0, max_value=127),
)
def test_pad_sequences_returns_tensor_with_correct_values(
    sequences: List[torch.Tensor], padding_value: int
) -> None:
    """Ensures Tensor returned by pad_sequences has correct values + padding."""
    out = pad_sequence(sequences, padding_value)

    # half doesn't support equality on CPU so cast to float
    if out.dtype == torch.float16:
        out = out.float()
        sequences = [seq.float() for seq in sequences]

    for i, seq in enumerate(sequences):
        seq_len = seq.size(-1)
        # check all tensor values (excluding padding)
        assert torch.all(out[i, ..., :seq_len] == seq)
        # check padding
        assert torch.all(out[i, ..., seq_len:] == padding_value)


# collate_fn ----------------------------------------


def test_seq_to_seq_collate_fn() -> None:
    """Unit test to ensure seq_to_seq_collate_fn returns correct values."""
    inputs = [rand([1, 2, 3]), rand([1, 2, 5])]
    seq_lens = tensor([3, 5])

    targets = [rand([10]), rand([7])]
    target_lengths = tensor([10, 7])

    batch = [
        ((inputs[0], seq_lens[0]), (targets[0], target_lengths[0])),
        ((inputs[1], seq_lens[1]), (targets[1], target_lengths[1])),
    ]

    x, y = seq_to_seq_collate_fn(batch)

    assert set(x.keys()) == {"x", "seq_lens"}
    assert set(y.keys()) == {"targets", "target_lengths"}

    assert torch.all(x["x"] == pad_sequence(inputs))
    assert torch.all(x["seq_lens"] == seq_lens)

    assert torch.all(y["targets"] == pad_sequence(targets))
    assert torch.all(y["target_lengths"] == target_lengths)
