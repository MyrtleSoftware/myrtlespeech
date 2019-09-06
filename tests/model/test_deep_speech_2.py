import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.model.deep_speech_2 import DeepSpeech2


# Fixtures and Strategies -----------------------------------------------------


class ModuleMock(torch.nn.Module):
    """Validates forward input type/size and outputs valid output type/size.

    Args:
        name: Name of module to mock.
        batch: Batch size.
        input_size: Expected size of input.
        output_size: Output size to generate.

    Attributes:
        called: A bool that is True if forward is called.
    """

    def __init__(self, name, batch, input_size, output_size):
        super().__init__()
        self.name = name
        self.batch = batch
        self.input_size = input_size
        self.output_size = output_size
        self.called = False

    def forward(self, x):
        self.called = True
        assert isinstance(x, tuple)
        assert len(x) == 2
        assert isinstance(x[0], torch.Tensor)
        assert x[0].size() == self.input_size
        assert x[0].requires_grad
        assert x[0].is_floating_point

        assert isinstance(x[1], torch.Tensor)
        assert x[1].size() == (self.batch,)
        assert not x[1].requires_grad

        out = torch.empty(
            self.output_size, dtype=torch.float32, requires_grad=True
        )
        out_seq_lens = torch.empty([self.batch])
        return out, out_seq_lens

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"batch={self.batch}, "
            f"input_size={self.input_size}, "
            f"output_size={self.output_size}, "
        )


@st.composite
def mock_ds2(draw) -> st.SearchStrategy[DeepSpeech2]:
    """Returns a strategy for DeepSpeech2 with valid ModuleMock modules."""
    # input size
    batch = draw(st.integers(min_value=1, max_value=32))
    input_channels = draw(st.integers(min_value=1, max_value=32))
    input_features = draw(st.integers(min_value=1, max_value=32))
    max_input_seq_len = draw(st.integers(min_value=1, max_value=32))

    # CNN
    cnn_channels = draw(st.integers(min_value=1, max_value=32))
    cnn_features = draw(st.integers(min_value=1, max_value=32))
    max_cnn_seq_len = draw(st.integers(min_value=1, max_value=32))
    cnn = ModuleMock(
        "cnn",
        batch,
        torch.Size([batch, input_channels, input_features, max_input_seq_len]),
        torch.Size([batch, cnn_channels, cnn_features, max_cnn_seq_len]),
    )

    # RNN
    max_rnn_seq_len = draw(st.integers(min_value=1, max_value=32))
    rnn_features = draw(st.integers(min_value=1, max_value=32))
    rnn = ModuleMock(
        "rnn",
        batch,
        torch.Size([max_cnn_seq_len, batch, cnn_channels * cnn_features]),
        output_size=torch.Size([max_rnn_seq_len, batch, rnn_features]),
    )

    # Lookahead
    lookahead = draw(
        st.one_of(st.none(), st.tuples(st.integers(1, 32), st.integers(1, 32)))
    )

    if lookahead is None:
        max_fc_in_seq_len = max_rnn_seq_len
        max_fc_in_features = rnn_features
        lookahead_module = None
    else:
        max_fc_in_seq_len = lookahead[0]
        max_fc_in_features = lookahead[1]
        lookahead_module = ModuleMock(
            "lookahead",
            batch,
            torch.Size([batch, rnn_features, max_rnn_seq_len]),
            torch.Size([batch, max_fc_in_features, max_fc_in_seq_len]),
        )

    # Fully connected
    max_out_seq_len = draw(st.integers(min_value=1, max_value=32))
    out_features = draw(st.integers(min_value=1, max_value=32))
    fully_connected = ModuleMock(
        "fully_connected",
        batch,
        torch.Size([batch, max_fc_in_seq_len, max_fc_in_features]),
        torch.Size([batch, max_out_seq_len, out_features]),
    )

    return DeepSpeech2(
        cnn=cnn,
        rnn=rnn,
        lookahead=lookahead_module,
        fully_connected=fully_connected,
    )


# Tests -----------------------------------------------------------------------


@given(ds2=mock_ds2())
def test_ds2_all_modules_called_with_valid_input_and_valid_output_returned(
    ds2: DeepSpeech2
) -> None:
    """Ensures all modules called with valid input + valid output returned."""
    input = (
        torch.empty(
            ds2.cnn.input_size, requires_grad=True, dtype=torch.float32
        ),
        torch.empty([ds2.cnn.batch], requires_grad=False, dtype=torch.long),
    )

    out = ds2(input)

    # check all modules called
    assert ds2.cnn.called
    assert ds2.rnn.called
    assert ds2.lookahead is None or ds2.lookahead.called
    assert ds2.fully_connected.called

    # check output type and size
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], torch.Tensor)
    output_size = ds2.fully_connected.output_size
    output_size = torch.Size([output_size[1], output_size[0], output_size[2]])
    assert out[0].size() == output_size
    assert out[0].requires_grad
    assert out[0].is_floating_point

    assert isinstance(out[1], torch.Tensor)
    assert out[1].size() == (ds2.cnn.batch,)
    assert not out[1].requires_grad
