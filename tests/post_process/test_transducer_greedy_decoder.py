import torch
from myrtlespeech.post_process.transducer_greedy_decoder import (
    TransducerGreedyDecoder,
)

from tests.post_process.utils import get_dummy_transducer


# Fixtures and Strategies -----------------------------------------------------


class TransducerGreedyDecoderDummy(TransducerGreedyDecoder):
    """A Decoder class which overrides :py:meth:`_joint_step` method."""

    def _joint_step(self, enc, pred):
        """Overrrides :py:meth:`_joint_step()`.

        This is necessary as :py:meth:`TransducerDecoderBase._joint_step()`
        performs a :py:meth:`log_softmax` which breaks this worked example.
        """
        logits, _ = self._model.joint_net((enc, pred))
        return logits.squeeze()


def get_fixed_decoder(max_symbols_per_step=100, blank_index=2):
    model = get_dummy_transducer(hidden_size=3)
    return TransducerGreedyDecoderDummy(
        blank_index=blank_index,
        model=model,
        max_symbols_per_step=max_symbols_per_step,
    )


# Tests -----------------------------------------------------------------------


def test_greedy_search_single_step(decoder=get_fixed_decoder()):
    """Worked example single step."""
    indata = torch.tensor([[[1.5, 0.0, 0.1]]])  # (1, 1, 3)
    indata = indata.unsqueeze(3)  # B, C, F, T = (1, 1, 3, 1)
    lengths = torch.IntTensor([1])
    inp = (indata, lengths)

    expected = [0]

    assert decoder.decode(inp)[0] == expected


def test_greedy_search_multi_step(decoder=get_fixed_decoder()):
    """Worked example multiple steps."""
    indata = torch.tensor(
        [[[[1.5, 0.0, 0.1], [0.3, 0.6, 0.1]]]]
    )  # (1, 1, 2, 3)
    indata = indata.transpose(2, 3)  # B, C, F, T = (1, 1, 3, 2)
    lengths = torch.IntTensor([2])

    expected = [0, 1, 0, 1]

    assert decoder.decode((indata, lengths))[0] == expected


def test_greedy_search_limit_symbols_per_step(
    decoder=get_fixed_decoder(max_symbols_per_step=1),
):
    """Worked example limit number of symbols."""
    indata = torch.tensor(
        [[[[1.5, 0.0, 0.1], [0.3, 0.6, 0.1]]]]
    )  # (1, 1, 2, 3)
    indata = indata.transpose(2, 3)  # B, C, F, T = (1, 1, 3, 2)
    lengths = torch.IntTensor([2])

    expected = [0, 1]

    assert decoder.decode((indata, lengths))[0] == expected


def test_multi_element_batch(decoder=get_fixed_decoder()):
    """Worked example multi element batch."""
    indata = torch.tensor(
        [
            [[[1.5, 0.0, 0.1], [0.3, 0.6, 0.1]]],
            [[[1.5, 0.0, 0.1], [0.3, 0.6, 0.1]]],
        ]
    )  # (2, 1, 2, 3)
    indata = indata.transpose(2, 3)
    lengths = torch.IntTensor([2, 1])

    expected = [[0, 1, 0, 1], [0]]

    assert decoder((indata, lengths))[0] == expected


def test_preserves_training_state(decoder=get_fixed_decoder()):
    """Checks training state is preserved."""
    indata = torch.tensor([[[1.5, 0.0, 0.1]]])  # (1, 1, 3)
    indata = indata.unsqueeze(3)  # B, C, F, T = (1, 1, 3, 1)
    lengths = torch.IntTensor([1])

    decoder._model.train()
    decoder((indata, lengths))
    assert decoder._model.training

    decoder._model.eval()
    decoder((indata, lengths))
    assert not decoder._model.training
