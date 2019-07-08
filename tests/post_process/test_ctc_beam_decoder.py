import torch

from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder


# Fixtures and Strategies -----------------------------------------------------


# Tests -----------------------------------------------------------------------

# Unit Tests ----------------------------------------


def test_simple_2x2() -> None:
    """Ensures simple beam search over 2x2 output beats naive greedy method.

    Let the output for a single example in a batch be:

        | symbol | p(symbol @ t = 0) | p(symbol @ t = 1) |
        |--------|-------------------|-------------------|
        | "a"    | 0.3               | 0.3               |
        | blank  | 0.7               | 0.7               |

    A naive approach will predict blank as the most likely sequence. The beam
    search should find p(a).

    Why?

    p(blank) = p(blank, blank)
             = 0.7 * 0.7
             = 0.49

    p(a) = p(a, blank) + p(blank, a) + p(a, a)
         = 0.21 + 0.21 + 0.09
         = 0.51
    """
    alphabet = {"a": 0, "_": 1}  # '_' represents "blank"
    # create tensor with grid described in docstring
    x = torch.empty((2, 1, len(alphabet)))  # seq_len=2, batch=1
    x[:, 0, alphabet["a"]] = torch.tensor([0.3, 0.3])
    x[:, 0, alphabet["_"]] = torch.tensor([0.7, 0.7])

    lengths = torch.tensor([2], dtype=torch.int8)

    ctc_decoder = CTCBeamDecoder(
        blank_index=alphabet["_"],
        beam_width=2,
        prune_threshold=0.0,
        separator_index=None,
        language_model=None,
    )

    result = ctc_decoder(x, lengths)

    assert result[0] == [alphabet["a"]]
