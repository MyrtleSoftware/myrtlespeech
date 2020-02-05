import inspect

import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.model.deep_speech_1 import DeepSpeech1


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def deep_speech_1s(
    draw, return_kwargs: bool = False
) -> st.SearchStrategy[DeepSpeech1]:
    kwargs = {
        "in_features": draw(st.integers(1, 128)),
        "n_hidden": draw(st.integers(1, 128)),
        "out_features": draw(st.integers(1, 128)),
        "drop_prob": draw(st.floats(0.0, 1.0, allow_nan=False)),
        "relu_clip": draw(st.floats(1.0, 20.0, allow_nan=False)),
        "forget_gate_bias": draw(st.floats(0.0, 1.0, allow_nan=False)),
    }

    # ensure all args are generated to catch modifications to class
    ds1_args = set(inspect.getfullargspec(DeepSpeech1.__init__).args[1:])
    assert ds1_args == set(kwargs.keys()), "DeepSpeech1 arguments missing"

    ds1 = DeepSpeech1(**kwargs)
    if return_kwargs:
        return ds1, kwargs
    return ds1


# Tests -----------------------------------------------------------------------


@given(st.data())
def test_all_gradients_computed_for_all_model_parameters(data) -> None:
    # create network
    ds1, kwargs = data.draw(deep_speech_1s(return_kwargs=True))

    # generate random input
    batch = data.draw(st.integers(1, 4))
    channels = 1
    features = kwargs["in_features"]
    seq_len = data.draw(st.integers(1, 8))

    x = torch.empty((batch, channels, features, seq_len)).normal_()
    seq_lens = torch.randint(
        low=1, high=seq_len + 1, size=(batch,), dtype=torch.long
    )

    # sort lengths since we require enforce_sorted=True
    seq_lens = seq_lens.sort(descending=True)[0]

    # forward pass
    out = ds1((x, seq_lens))

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    # check all parameters have gradient
    for name, p in ds1.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
