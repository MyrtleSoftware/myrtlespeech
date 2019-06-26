import torch
from hypothesis import given

from myrtlespeech.model.encoder.encoder import conv_to_rnn_size
from tests.utils.utils import tensors


# Tests -----------------------------------------------------------------------


@given(tensor=tensors(min_n_dims=4, max_n_dims=4))
def test_conv_to_rnn_size_returns_tensor_with_correct_size(
    tensor: torch.Tensor
) -> None:
    """Ensures the Tensor returned by conv_to_rnn_size has correct size."""
    batch, channels, features, seq_len = tensor.size()
    out = conv_to_rnn_size(tensor)
    out_batch, out_seq_len, out_features = out.size()

    assert batch == out_batch
    assert seq_len == out_seq_len
    assert channels * features == out_features
