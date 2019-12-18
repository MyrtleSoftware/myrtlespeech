import hypothesis.strategies as st
import torch
from hypothesis import given
from hypothesis import settings
from myrtlespeech.model.fully_connected import FullyConnected
from myrtlespeech.model.rnn_t import RNNTJointNet

# Tests -----------------------------------------------------------------------


@given(
    data=st.data(),
    batch=st.integers(min_value=1, max_value=32),
    max_seq_len=st.integers(min_value=2, max_value=12),
    max_label_length=st.integers(min_value=1, max_value=5),
    pred_net_out_feat=st.integers(min_value=1, max_value=5),
    encoder_out_feat=st.integers(min_value=1, max_value=5),
    vocab_size=st.integers(min_value=1, max_value=5),
    fc_hid_layers=st.integers(min_value=1, max_value=3),
    activation_fn=st.sampled_from([torch.nn.ReLU(), torch.nn.Tanh()]),
)
@settings(deadline=4000)
def test_joint_network_memory_efficient_equivalent_to_standard_path(
    data,
    batch: int,
    max_seq_len: int,
    max_label_length: int,
    pred_net_out_feat: int,
    encoder_out_feat: int,
    vocab_size: int,
    fc_hid_layers: int,
    activation_fn: torch.nn.Module,
) -> None:
    """Tests that two paths (memory_efficient and not) are equivalent."""
    fc = FullyConnected(
        in_features=encoder_out_feat + pred_net_out_feat,
        out_features=vocab_size + 1,
        hidden_size=vocab_size + 1,
        num_hidden_layers=fc_hid_layers,
        hidden_activation_fn=activation_fn,
    )

    model_mem_efficient = RNNTJointNet(fc=fc, memory_efficient=True)
    model_standard = RNNTJointNet(fc=fc, memory_efficient=False)

    # generate random inputs f and g where:
    f = torch.empty((max_seq_len, batch, encoder_out_feat)).normal_()
    f_lens = torch.randint(
        low=1, high=max_seq_len, size=(batch,), dtype=torch.long
    )
    g = torch.empty((batch, max_label_length + 1, pred_net_out_feat)).normal_()
    g_lens = torch.randint(
        low=1, high=max_label_length + 1, size=(batch,), dtype=torch.long
    )
    # ensure max values are present in lengths
    f_lens[0] = max_seq_len
    g_lens[0] = max_label_length + 1

    if torch.cuda.is_available():
        input = (f.cuda(), f_lens.cuda()), (g.cuda(), g_lens.cuda())
    else:
        input = (f, f_lens), (g, g_lens)

    # forward pass
    res1 = model_mem_efficient(input)
    res2 = model_standard(input)

    expected_shape = (batch, max_seq_len, max_label_length + 1, vocab_size + 1)

    assert res1[0].shape == res2[0].shape == expected_shape
    assert torch.allclose(res1[0], res2[0])
    assert torch.allclose(res1[1], res2[1])
