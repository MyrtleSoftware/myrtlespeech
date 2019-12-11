import math

import hypothesis.strategies as st
import torch
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.transducer import build as build_transducer

from tests.protos.test_transducer import transducer


# Tests -----------------------------------------------------------------------


@given(
    data=st.data(),
    input_features=st.integers(min_value=2, max_value=12),
    input_channels=st.integers(min_value=1, max_value=5),
    vocab_size=st.integers(min_value=2, max_value=32),
    time_reduction=st.booleans(),
)
@settings(deadline=5000)
def test_all_gradients_computed_for_all_parameters_and_size_as_expected(
    data,
    input_features: int,
    input_channels: int,
    vocab_size: int,
    time_reduction: bool,
) -> None:
    """Tests that gradients are computed and output shape is as expected."""
    # create network
    transducer_cfg, kwargs = data.draw(
        transducer(time_reduction=time_reduction, return_kwargs=True)
    )
    model = build_transducer(
        transducer_cfg, input_features, input_channels, vocab_size
    )

    # generate random input
    batch = data.draw(st.integers(1, 4))
    seq_len = data.draw(st.integers(3, 8))
    label_seq_len = data.draw(st.integers(2, 6))

    x = torch.empty((batch, input_channels, input_features, seq_len)).normal_()
    seq_lens = torch.randint(
        low=1, high=seq_len, size=(batch,), dtype=torch.long
    )
    y = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch, label_seq_len),
        dtype=torch.long,
    )
    label_seq_lens = torch.randint(
        low=1, high=label_seq_len, size=(batch,), dtype=torch.long
    )
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

    # forward pass
    out = model(input)

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    expected_seq_len = seq_len
    if time_reduction:
        factor = kwargs["transducer_encoder"]["time_reduction_factor"]
        expected_seq_len = math.ceil(seq_len / factor)
    expected_shape = (
        batch,
        expected_seq_len,
        label_seq_len + 1,
        vocab_size + 1,
    )

    assert out[0].shape == expected_shape

    # check all parameters have gradient
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
