from typing import Dict

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from myrtlespeech.loss.ctc_loss import CTCLoss

from tests.utils.utils import arrays
from tests.utils.utils import tensors


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def ctc_loss_arguments(draw) -> st.SearchStrategy[Dict]:
    """Generates args for class constructor and forward."""
    ret_args = {}

    # generate input tensor
    ret_args["inputs"] = draw(
        tensors(min_n_dims=3, max_n_dims=3, min_dim_size=2, max_dim_size=32)
    )
    # get shapes, convert to Python ints for Hypothesis
    max_seq_len, batch, features = ret_args["inputs"].size()
    max_seq_len = int(max_seq_len)
    batch = int(batch)
    features = int(features)

    # generate CTCLoss arguments
    ret_args["blank"] = draw(st.integers(min_value=0, max_value=features - 1))
    ret_args["reduction"] = draw(st.sampled_from(["none", "mean", "sum"]))
    ret_args["zero_infinity"] = draw(st.booleans())

    # generate remaining CTCLoss.forward arguments
    ret_args["targets"] = torch.tensor(
        draw(
            arrays(
                shape=(batch, max_seq_len),
                dtype=np.int32,
                elements=st.integers(
                    min_value=0, max_value=features - 1
                ).filter(lambda x: x != ret_args["blank"]),
            )
        ),
        requires_grad=False,
    )

    ret_args["input_lengths"] = torch.tensor(
        draw(
            arrays(
                shape=(batch,),
                dtype=np.int32,
                elements=st.integers(min_value=1, max_value=max_seq_len),
            )
        ),
        requires_grad=False,
    )

    target_lengths = []
    for length in ret_args["input_lengths"]:
        # ensure CTC requirement that target length <= input length
        target_lengths.append(draw(st.integers(1, int(length))))
    ret_args["target_lengths"] = torch.tensor(
        target_lengths, dtype=torch.int32, requires_grad=False
    )

    return ret_args


# Tests -----------------------------------------------------------------------


@given(args=ctc_loss_arguments())
def test_ctc_loss_matches_torch(args) -> None:
    """Ensures CTCLoss matches torch CTCLoss(LogSoftmax(...), ...)."""
    myrtle_ctc_loss = CTCLoss(
        blank=args["blank"],
        reduction=args["reduction"],
        zero_infinity=args["zero_infinity"],
    )

    torch_log_softmax = torch.nn.LogSoftmax(dim=-1)
    torch_ctc_loss = torch.nn.CTCLoss(
        blank=args["blank"],
        reduction=args["reduction"],
        zero_infinity=args["zero_infinity"],
    )

    actual = myrtle_ctc_loss(
        inputs=(args["inputs"], args["input_lengths"]),
        targets=(args["targets"], args["target_lengths"]),
    )

    expected = torch_ctc_loss(
        log_probs=torch_log_softmax(args["inputs"]),
        targets=args["targets"],
        input_lengths=args["input_lengths"],
        target_lengths=args["target_lengths"],
    )

    assert isinstance(actual, torch.Tensor)
    assert isinstance(expected, torch.Tensor)

    assert actual.size() == expected.size()

    assert torch.allclose(actual, expected)
