from typing import Union

from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder


class SpeechToText(SeqToSeq):
    """A :py:class:`.SeqToSeq` model for speech recognition.

    Args:
        alphabet: A :py:class:`.Alphabet` for converting symbols to integers
            and vice versa.

        post_process: An optional decoder.

        args/kwargs: See :py:class:`.SeqToSeq`.
    """

    def __init__(
        self,
        alphabet: Alphabet,
        post_process: Union[None, CTCGreedyDecoder, CTCBeamDecoder],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alphabet = alphabet
        self.post_process = post_process

    def extra_repr(self) -> str:
        return f"(alphabet): {self.alphabet}"
