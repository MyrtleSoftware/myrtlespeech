from typing import Union

from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder


class SpeechToText(SeqToSeq):
    """TODO"""

    def __init__(
        self,
        alphabet: Alphabet,
        post_process: Union[None, CTCGreedyDecoder, CTCBeamDecoder],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alphabet = alphabet
