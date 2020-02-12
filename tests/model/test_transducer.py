from typing import Tuple

import hypothesis.strategies as st
import torch
from hypothesis import assume
from hypothesis import given
from myrtlespeech.builders.transducer import build as build_transducer
from myrtlespeech.model.transducer import Transducer

from tests.protos.test_transducer import transducers
from tests.utils.utils import tensors

# Utilities -------------------------------------------------------------------
@st.composite
def transducers_and_valid_inputs(
    draw,
) -> Tuple[
    st.SearchStrategy[Transducer],
    st.SearchStrategy[
        Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ],
    st.SearchStrategy[int],
]:
    """Returns a Transducer with valid inputs + vocab_size."""
    x = draw(tensors(min_n_dims=4, max_n_dims=4))
    (batch, input_channels, input_features, seq_len) = x.size()
    assume(seq_len > 1)
    vocab_size = draw(st.integers(min_value=2, max_value=32))
    label_seq_len = draw(st.integers(2, 6))
    transducer_cfg = draw(transducers())

    transducer = build_transducer(
        transducer_cfg, input_features, input_channels, vocab_size
    )

    seq_lens = torch.randint(
        low=1, high=seq_len, size=(batch,), dtype=torch.long
    )
    # sort lengths since we require enforce_sorted=True
    seq_lens = seq_lens.sort(descending=True)[0]

    y = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch, label_seq_len),
        dtype=torch.long,
    )
    label_seq_lens = torch.randint(
        low=1, high=label_seq_len, size=(batch,), dtype=torch.long
    )

    # sort lengths since we require enforce_sorted=True
    label_seq_lens = label_seq_lens.sort(descending=True)[0]

    # ensure max values are present in lengths
    seq_lens[0] = seq_len
    label_seq_lens[0] = label_seq_len

    if torch.cuda.is_available():
        input = (
            (x.cuda(), seq_lens.cuda()),
            (y.cuda(), label_seq_lens.cuda()),
        )
    else:
        input = ((x, seq_lens), (y, label_seq_lens))

    return transducer, input, vocab_size


# Tests -----------------------------------------------------------------------


@given(transducers_and_valid_inputs())
def test_all_gradients_computed_for_all_parameters_and_size_as_expected(
    transducers_and_valid_inputs: Tuple[
        Transducer,
        Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        int,
    ]
) -> None:
    """Tests that gradients are computed and output shape is as expected."""
    # create network
    transducer, input, vocab_size = transducers_and_valid_inputs
    (batch, _, _, seq_len) = input[0][0].size()
    _, label_seq_len = input[1][0].size()

    # check generation
    assert len(input[0][1]) == len(input[1][1]) == batch

    # forward pass
    out = transducer(input)

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    assert out[0].shape == (batch, seq_len, label_seq_len + 1, vocab_size + 1)

    # check all parameters have gradient
    for name, p in transducer.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
