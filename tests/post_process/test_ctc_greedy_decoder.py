from typing import Callable, Tuple, List

import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder
from tests.data.test_alphabet import random_alphabet


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def decoder_input_outputs(
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
    lengths = torch.tensor(input_lengths)

    return blank_index, (x, lengths), sentences


# Tests -----------------------------------------------------------------------


@given(input_output=decoder_input_outputs())
def test_ctc_greedy_decoder_correct_decode(input_output) -> None:
    blank_index, (x, lengths), exp_sentences = input_output
    ctc_decoder = CTCGreedyDecoder(blank_index)

    act_sentences = ctc_decoder(x, lengths)

    assert act_sentences == exp_sentences
