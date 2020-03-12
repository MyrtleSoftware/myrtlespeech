import inspect
from typing import Dict
from typing import Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.model.deep_speech_1 import DeepSpeech1
from myrtlespeech.model.hard_lstm import HardLSTM
from myrtlespeech.model.rnn import RNN


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def deep_speech_1s(
    draw, return_kwargs: bool = False
) -> st.SearchStrategy[DeepSpeech1]:
    kwargs = {
        "input_features": draw(st.integers(1, 32)),
        "input_channels": draw(st.integers(1, 4)),
        "n_hidden": draw(st.integers(1, 128)),
        "out_features": draw(st.integers(1, 128)),
        "drop_prob": draw(st.floats(0.0, 1.0, allow_nan=False)),
        "relu_clip": draw(st.floats(1.0, 20.0, allow_nan=False)),
        "forget_gate_bias": draw(st.floats(0.0, 1.0, allow_nan=False)),
        "hard_lstm": draw(st.booleans()),
    }

    # ensure all args are generated to catch modifications to class
    ds1_args = set(inspect.getfullargspec(DeepSpeech1.__init__).args[1:])
    assert ds1_args == set(kwargs.keys()), "DeepSpeech1 arguments missing"

    ds1 = DeepSpeech1(**kwargs)
    if return_kwargs:
        return ds1, kwargs
    return ds1


# Tests -----------------------------------------------------------------------


@given(deep_speech_1s(return_kwargs=True))
def test_correct_ds1_returned(ds1_kwargs: Tuple[DeepSpeech1, Dict]) -> None:
    """Ensures correct ``rnn`` type and initialisation."""
    ds1, kwargs = ds1_kwargs
    fc1 = ds1.fc1
    fc2 = ds1.fc2
    fc3 = ds1.fc3
    fc4 = ds1.fc4
    fcout = ds1.out

    if not isinstance(fc1, torch.nn.Linear):
        fc1 = fc1[0]
    if not isinstance(fc2, torch.nn.Linear):
        fc2 = fc2[0]
    if not isinstance(fc3, torch.nn.Linear):
        fc3 = fc3[0]
    if not isinstance(fc4, torch.nn.Linear):
        fc4 = fc4[0]
    if not isinstance(fcout, torch.nn.Linear):
        fcout = fcout[0]
    assert (
        fc1.in_features == kwargs["input_features"] * kwargs["input_channels"]
    )
    assert (
        fc1.out_features
        == fc2.in_features
        == fc2.out_features
        == fc3.in_features
        == fc4.out_features
        == fcout.in_features
        == kwargs["n_hidden"]
    )
    assert fc3.out_features == fc4.in_features == 2 * kwargs["n_hidden"]
    assert fcout.out_features == kwargs["out_features"]
    if kwargs["hard_lstm"]:
        assert isinstance(ds1.bi_lstm, HardLSTM)
    else:
        assert isinstance(ds1.bi_lstm, RNN)


@given(st.data())
def test_all_gradients_computed_for_all_model_parameters(data) -> None:
    # create network
    ds1, kwargs = data.draw(deep_speech_1s(return_kwargs=True))

    # generate random input
    batch = data.draw(st.integers(1, 4))
    channels = kwargs["input_channels"]
    features = kwargs["input_features"]
    seq_len = data.draw(st.integers(1, 8))

    x = torch.empty((batch, channels, features, seq_len)).normal_()
    seq_lens = torch.randint(
        low=1, high=seq_len + 1, size=(batch,), dtype=torch.long
    )

    # sort lengths since we require enforce_sorted=True
    seq_lens = seq_lens.sort(descending=True)[0]

    # forward pass
    out, _ = ds1((x, seq_lens))

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    # check all parameters have gradient
    for name, p in ds1.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
