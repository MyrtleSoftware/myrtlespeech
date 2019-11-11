from typing import Tuple

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from myrtlespeech.post_process.rnn_t_beam_decoder import RNNTBeamDecoder

# Fixtures and Strategies -----------------------------------------------------
class RNN:
    def __init__(self, hidden_size=3):
        self.hidden_size = hidden_size


class DecRNN:
    def __init__(self, hidden_size=3):
        self.rnn = RNN(hidden_size)

    def __call__(self, x):
        """TODO: The way this works is by using the hidden state to decide which
        chars to upweight at a given timestep.
         ``[B, U + 1, H]``"""

        if isinstance(x[0], torch.Tensor):
            embedded = x[0]
            state = None
            return_tuple = False
        elif isinstance(x[0], tuple) and len(x[0]) == 2:
            embedded, state = x[0]
            return_tuple = True
        else:
            raise ValueError(
                "`x[0]` must be of form (input, hidden) or (input)."
            )

        B, U_, H = embedded.shape

        assert B == U_ and B == 1, "Currently only supports batch=seq len == 1"

        lengths = x[1]

        if state is None:
            # `state` is the index of char to upweight.
            # In first instance upweight char at index 1
            state = 1
        if embedded.squeeze().int()[0] == 0:
            # i.e. if this is the SOS,
            # assign probability of 0.2 to all three chars:
            res = torch.ones(3, dtype=torch.float32) * 0.2
        else:
            res = torch.ones(3, dtype=torch.float32) * 0.1
            state_to_upweight = embedded.squeeze().int()[1]
            res[state_to_upweight] += 0.3

        res[state] += 0.4
        out, hid = torch.log(res), (state + 1) % 3

        # blow up to full dimension
        out = out.unsqueeze(0).unsqueeze(0)

        if return_tuple:
            return (out, hid), lengths
        else:
            return out, lengths


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = False
        self.dec_rnn = DecRNN(3)

    def forward(self):
        raise NotImplementedError

    def encode(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ``[B, C, H, T]`` -> ``[T, B, H_out]`` """

        h = x[0]  # B, C, H, T
        B, C, F, T = h.shape

        assert C == 1, f"In DummyModel(), input channels must == 1 but C == {C}"
        h = h.squeeze(1)  # B, H, T
        h = h.permute(2, 0, 1)

        return h, x[1]

    def embedding(self, x):
        res = torch.tensor([1, x[0].item() + 1]).unsqueeze(0).unsqueeze(0)
        x = res, x[1]

        return x

    # def prediction(self, y, state=None, add_sos=False):
    #     assert add_sos == False, "This test assumes add_sos=False"
    #     if y is not None:
    #         y = self.embedding(y)
    #     else:
    #         B = 1 if state is None else state[0].size(1)
    #         y = torch.zeros((1, B, 2))
    #
    #     input = ((y, state), lengths)
    #     return self.dec_rnn(input)


class RNNTBeamDecoderDummy(RNNTBeamDecoder):
    """Decoder class which overrides _joint_step method"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _joint_step(self, enc, pred):
        """Overrride _joint_step() as it performs a log on outputs.

        TODO: remove this class and perform an exp on inputs instead. """

        f, f_lens = enc
        g, g_lens = pred
        seq_lengths = (f_lens, g_lens)

        T, B1, H1 = f.shape
        B2, U_, H2 = g.shape

        assert (
            B1 == B2
        ), "Batch size from prediction network and transcription must be equal"

        f = f.transpose(1, 0)  # (T, B, H1) -> (B, T, H1)
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)

        g = g.unsqueeze(dim=1)  # (B, 1, U_, H)
        print("f", f)
        print("g", g)
        print("joint", (f + g).squeeze())
        print()
        return (f + g).squeeze()


@pytest.fixture
def decoder():
    alphabet = ["_", "a", "b"]
    blank_index = 0
    model = DummyModel()
    length_norm = False
    max_symbols_per_step = 100
    return RNNTBeamDecoderDummy(
        blank_index=blank_index,
        model=model,
        beam_width=2,
        length_norm=length_norm,
        max_symbols_per_step=max_symbols_per_step,
    )


# Tests -----------------------------------------------------------------------


def test_beam_search_single_step(decoder):
    indata = torch.tensor([[[-3.0, -0.2, -1.9]]])
    indata = indata.unsqueeze(3)  # B, C, F, T = (1, 1, 3, 1)
    lengths = torch.IntTensor([1])
    inp = (indata, lengths)
    assert decoder.decode(inp) == [1, 1]


#
# def test_simple_2x2() -> None:
#     """Ensures simple beam search over 2x2 output beats naive greedy method.
#
#     Let the output for a single example in a batch be:
#
#         | symbol | p(symbol @ t = 0) | p(symbol @ t = 1) |
#         |--------|-------------------|-------------------|
#         | "a"    | 0.3               | 0.3               |
#         | blank  | 0.7               | 0.7               |
#
#     A naive approach will predict blank as the most likely sequence. The beam
#     search should find p(a).
#
#     Why?
#
#     p(blank) = p(blank, blank)
#              = 0.7 * 0.7
#              = 0.49
#
#     p(a) = p(a, blank) + p(blank, a) + p(a, a)
#          = 0.21 + 0.21 + 0.09
#          = 0.51
#     """
#     alphabet = {"a": 0, "_": 1}  # '_' represents "blank"
#     # create tensor with grid described in docstring
#     x = torch.empty((2, 1, len(alphabet)))  # seq_len=2, batch=1
#     x[:, 0, alphabet["a"]] = torch.tensor([0.3, 0.3])
#     x[:, 0, alphabet["_"]] = torch.tensor([0.7, 0.7])
#
#     lengths = torch.tensor([2], dtype=torch.int8)
#
#     ctc_decoder = CTCBeamDecoder(
#         blank_index=alphabet["_"],
#         beam_width=2,
#         prune_threshold=0.0,
#         separator_index=None,
#         language_model=None,
#     )
#
#     result = ctc_decoder(x, lengths)
#
#     assert result[0] == [alphabet["a"]]
#
#
# def test_simple_lanaguage_model() -> None:
#     """Ensures beam search with language model corrects homophones.
#
#     Without a language model the test should find "do" as the most likely. We
#     create two language models that return 2.0 (invalid probability, yes) for
#     both "dew" and "due" respectively and 0.0 for all other words.
#
#     The test ensures that "dew" and "due" are returned when their respective
#     language models are used.
#     """
#     alphabet = dict(zip("deouw_ ", range(7)))
#     x = torch.empty((4, 1, len(alphabet)))
#     x[:, 0, alphabet["d"]] = torch.tensor([0.75, 0.05, 0.10, 0.01])
#     x[:, 0, alphabet["e"]] = torch.tensor([0.05, 0.20, 0.20, 0.01])
#     x[:, 0, alphabet["o"]] = torch.tensor([0.05, 0.30, 0.35, 0.01])
#     x[:, 0, alphabet["u"]] = torch.tensor([0.05, 0.20, 0.10, 0.01])
#     x[:, 0, alphabet["w"]] = torch.tensor([0.05, 0.00, 0.20, 0.01])
#     x[:, 0, alphabet["_"]] = torch.tensor([0.00, 0.00, 0.10, 0.94])
#     x[:, 0, alphabet[" "]] = torch.tensor([0.05, 0.05, 0.05, 0.01])
#
#     lengths = torch.tensor([x.size()[0]], dtype=torch.int8)
#
#     def lm(target: str):
#         target_tuple = tuple(alphabet[c] for c in target) + (alphabet[" "],)
#         return lambda w: 2.0 if w == target_tuple else 0.0
#
#     # first check no LM returns "do"
#     ctc_decoder = CTCBeamDecoder(blank_index=alphabet["_"], beam_width=20)
#     assert ctc_decoder(x, lengths) == [[alphabet[c] for c in "do"]]
#
#     # check LM corrects this to dew or due
#     for target in ["dew", "due"]:
#         ctc_decoder = CTCBeamDecoder(
#             blank_index=alphabet["_"],
#             beam_width=20,
#             separator_index=alphabet[" "],
#             language_model=lm(target),
#             lm_weight=10.0,
#             word_weight=2.0,
#         )
#
#         assert ctc_decoder(x, lengths) == [[alphabet[c] for c in target + " "]]
#
#
# def test_prediction_returned_for_each_element_in_batch() -> None:
#     """Ensures a prediction is returned for each element in the batch."""
#     seq_len, n_batch, symbols = 10, 5, 30
#     batch = torch.empty([seq_len, n_batch, symbols], requires_grad=False)
#     batch = torch.nn.functional.softmax(batch, dim=-1)
#     lengths = torch.tensor([8, 3, 9, 7, 2])
#
#     ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=8)
#
#     assert len(ctc_decoder(batch, lengths)) == n_batch
#
#
# @given(dtype=st.sampled_from([torch.half, torch.float, torch.double]))
# def test_ctc_beam_decoder_raises_value_error_for_float_dtypes(
#     dtype: torch.dtype
# ) -> None:
#     """Ensures ValueError raised when lengths.dtype is float."""
#     x = torch.empty((5, 1, 3))
#     lengths = torch.tensor([5], dtype=dtype)
#     ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=20)
#     with pytest.raises(ValueError):
#         ctc_decoder(x, lengths)
#
#
# @given(x_batch_size=st.integers(1, 32), lengths_batch_size=st.integers(1, 32))
# def test_ctc_beam_decoder_raises_value_error_when_batch_x_lengths_differ(
#     x_batch_size: int, lengths_batch_size: int
# ) -> None:
#     """Ensures ValueError raised when batch size of x and lengths differs."""
#     assume(x_batch_size != lengths_batch_size)
#
#     ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=20)
#
#     # create input tensors, batch and alphabet size fixed to 10 and 5
#     x = torch.empty((10, x_batch_size, 5))
#     lengths = torch.empty(lengths_batch_size, dtype=torch.int16)
#
#     with pytest.raises(ValueError):
#         ctc_decoder(x, lengths)
#
#
# def test_ctc_beam_decoder_raises_value_error_lengths_values_greater_seq_len() -> None:
#     """Ensures ValueError when lengths entry is greater than seq len of x."""
#     ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=20)
#
#     x = torch.empty((10, 3, 20))
#     lengths = torch.tensor([1, 19, 1], dtype=torch.uint8)
#
#     with pytest.raises(ValueError):
#         ctc_decoder(x, lengths)
#
#
# @given(blank_index=st.integers(-1000, -1))
# def test_ctc_beam_decoder_raises_value_error_negative_blank_index(
#     blank_index: int
# ) -> None:
#     """Ensures ValueError raised when blank_index < 0."""
#     with pytest.raises(ValueError):
#         CTCBeamDecoder(blank_index=blank_index, beam_width=20)
#
#
# @given(beam_width=st.integers(-1000, 0))
# def test_ctc_beam_decoder_raises_value_error_non_positive_beam_width(
#     beam_width: int
# ) -> None:
#     """Ensures ValueError raised when beam_width <= 0."""
#     with pytest.raises(ValueError):
#         CTCBeamDecoder(blank_index=0, beam_width=beam_width)
#
#
# @given(
#     prune_threshold=st.one_of(
#         st.floats(max_value=-0.0, exclude_max=True),
#         st.floats(min_value=1.0, exclude_min=True),
#     )
# )
# def test_ctc_beam_decoder_raises_value_error_prune_threshold_not_0_1(
#     prune_threshold: float
# ) -> None:
#     """Ensures ValueError raised when prune_threshold not in [0.0, 1.0]."""
#     with pytest.raises(ValueError):
#         CTCBeamDecoder(
#             blank_index=0, beam_width=1, prune_threshold=prune_threshold
#         )
#
#
# def test_ctc_beam_decoder_raises_value_error_lm_but_no_lm_weight() -> None:
#     """Ensures ValueError raised when language_model set but not lm_weight."""
#     with pytest.raises(ValueError):
#         CTCBeamDecoder(
#             blank_index=0, beam_width=20, language_model=lambda _: 0.0
#         )
#
#
# @given(separator_index=st.integers(-1000, -1))
# def test_ctc_beam_decoder_raises_value_error_negative_separator_index(
#     separator_index: int
# ) -> None:
#     """Ensures ValueError raised when separator_index < 0."""
#     with pytest.raises(ValueError):
#         CTCBeamDecoder(
#             blank_index=0, beam_width=20, separator_index=separator_index
#         )
