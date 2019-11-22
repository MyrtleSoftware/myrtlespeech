import hypothesis.strategies as st
import torch
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.rnn_t import build as build_rnn_t
from myrtlespeech.protos import rnn_t_pb2

from tests.protos.test_rnn_t import rnn_t


# Tests -----------------------------------------------------------------------


@given(
    data=st.data(),
    rnn_t_cfg=rnn_t(),
    input_features=st.integers(min_value=2, max_value=12),
    input_channels=st.integers(min_value=1, max_value=5),
    vocab_size=st.integers(min_value=2, max_value=32),
)
@settings(deadline=5000)
def test_all_gradients_computed_for_all_model_parameters(
    data,
    rnn_t_cfg: rnn_t_pb2.RNNT,
    input_features: int,
    input_channels: int,
    vocab_size: int,
) -> None:

    # create network
    rnnt = build_rnn_t(rnn_t_cfg, input_features, input_channels, vocab_size)

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
            (x.cuda(), y.cuda()),
            (seq_lens.cuda(), label_seq_lens.cuda()),
        )
    else:
        input = ((x, y), (seq_lens, label_seq_lens))

    # forward pass
    out = rnnt(input)

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    # check all parameters have gradient
    for name, p in rnnt.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
