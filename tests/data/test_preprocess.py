import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.data.preprocess import AddSequenceLength

from tests.utils.utils import tensors


# Fixtures and Strategies -----------------------------------------------------


# Tests -----------------------------------------------------------------------


@given(data=st.data(), tensor=tensors(min_n_dims=1))
def test_add_sequence_length_returns_correct_seq_len(
    data, tensor: torch.Tensor
) -> None:
    """Ensures AddSequenceLength returns correct sequence length."""
    length_dim = data.draw(
        st.integers(min_value=0, max_value=len(tensor.size()) - 1)
    )

    add_seq_len = AddSequenceLength(length_dim=length_dim)

    out, seq_len = add_seq_len(tensor)

    assert torch.all(out == tensor)
    assert seq_len == torch.tensor([tensor.size(length_dim)])
