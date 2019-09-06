from typing import Tuple

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder

from tests.data.test_alphabet import random_alphabet


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def ctc_greedy_decoder_input_outputs(
    draw,
) -> st.SearchStrategy[
    Tuple[
        int,  # blank index
        Tuple[torch.Tensor, torch.Tensor],  # x, lengths
        Tuple[torch.Tensor, torch.Tensor],  # output, output_lengths
    ]
]:
    """Returns a SearchStrategy for (blank_index, input, expected output)."""
    alphabet = draw(random_alphabet())
    assume(len(alphabet) > 1)  # must be at least blank and one other symbol
    blank_index = draw(st.integers(0, len(alphabet) - 1))

    # generate random batch of sentence (indices) excluding blank index
    batch_size = draw(st.integers(1, 8))
    non_blanks = alphabet.get_indices(list(alphabet))
    non_blanks.pop(blank_index)
    sentences = [
        draw(st.lists(st.sampled_from(non_blanks), min_size=1))
        for _ in range(batch_size)
    ]

    # for each sentence insert "blank" between duplicate symbols and replicate
    # some symbols
    blank_sentences = []
    for sentence in sentences:
        blank_sentence = []
        prev = None
        for symbol_idx in sentence:
            if prev is not None and prev == symbol_idx:
                n_rep = draw(st.integers(1, 5))
                blank_sentence.extend([blank_index] * n_rep)

            n_rep = draw(st.integers(1, 5))
            blank_sentence.extend([symbol_idx] * n_rep)

            prev = symbol_idx

        blank_sentences.append(blank_sentence)

    # compute inputs
    longest = max([len(sentence) for sentence in blank_sentences])
    input_sentences = []  # list of input 2D tensors (longest, len(alphabet))
    input_lengths = []  # list of input lengths
    for sentence in blank_sentences:
        input_sentence = torch.empty((longest, len(alphabet))).normal_()

        # ensure desired symbol has greatest value at each time step by summing
        # up abs value of all symbols
        for seq_idx, sym_idx in enumerate(sentence):
            input_sentence[seq_idx, sym_idx] = (
                0.5 + input_sentence[seq_idx, :].abs().sum()
            )

        input_sentences.append(input_sentence)
        input_lengths.append(len(sentence))

    x = torch.stack(input_sentences, dim=1)

    supported_dtypes = [torch.int64]
    if longest <= 2 ** 31 - 1:
        supported_dtypes.append(torch.int32)
    if longest <= 2 ** 15 - 1:
        supported_dtypes.append(torch.int16)
    if longest <= 2 ** 8 - 1:
        supported_dtypes.append(torch.uint8)
    if longest <= 2 ** 7 - 1:
        supported_dtypes.append(torch.int8)
    lengths_dtype = draw(st.sampled_from(supported_dtypes))
    lengths = torch.tensor(input_lengths, dtype=lengths_dtype)

    return blank_index, (x, lengths), sentences


# Tests -----------------------------------------------------------------------


@given(input_output=ctc_greedy_decoder_input_outputs())
def test_ctc_greedy_decoder_correct_decode(input_output) -> None:
    blank_index, (x, lengths), exp_sentences = input_output
    ctc_decoder = CTCGreedyDecoder(blank_index)

    act_sentences = ctc_decoder(x, lengths)

    assert act_sentences == exp_sentences


@given(
    input_output=ctc_greedy_decoder_input_outputs(),
    dtype=st.sampled_from([torch.half, torch.float, torch.double]),
)
def test_ctc_greedy_decoder_raises_value_error_for_float_dtypes(
    input_output, dtype: torch.dtype
) -> None:
    """Ensures ValueError raised when lengths.dtype is float."""
    blank_index, (x, lengths), exp_sentences = input_output
    lengths = lengths.to(dtype)
    ctc_decoder = CTCGreedyDecoder(blank_index)
    with pytest.raises(ValueError):
        ctc_decoder(x, lengths)


@given(x_batch_size=st.integers(1, 32), lengths_batch_size=st.integers(1, 32))
def test_ctc_greedy_decoder_raises_value_error_when_batch_x_lengths_differ(
    x_batch_size: int, lengths_batch_size: int
) -> None:
    """Ensures ValueError raised when batch size of x and lengths differs."""
    assume(x_batch_size != lengths_batch_size)

    ctc_decoder = CTCGreedyDecoder(0)

    # create input tensors, batch and alphabet size fixed to 10 and 5
    x = torch.empty((10, x_batch_size, 5))
    lengths = torch.empty(lengths_batch_size, dtype=torch.int16)

    with pytest.raises(ValueError):
        ctc_decoder(x, lengths)


@given(data=st.data(), input_output=ctc_greedy_decoder_input_outputs())
def test_ctc_greedy_decoder_raises_value_error_lengths_values_greater_seq_len(
    data, input_output
) -> None:
    """Ensures ValueError when lengths entry is greater than seq len of x."""
    blank_index, (x, lengths), exp_sentences = input_output
    seq_len, batch, _ = x.size()
    ctc_decoder = CTCGreedyDecoder(blank_index)

    invalid_length = data.draw(st.integers(seq_len + 1, 3 * seq_len))
    assume(invalid_length <= torch.iinfo(lengths.dtype).max)
    invalid_idx = data.draw(st.integers(0, batch - 1))
    lengths[invalid_idx] = invalid_length

    with pytest.raises(ValueError):
        ctc_decoder(x, lengths)
