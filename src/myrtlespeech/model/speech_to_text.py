from typing import Callable, List, Tuple, Union

import torch

from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.model.encoder_decoder import EncoderDecoder
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder


class SpeechToText:
    """A speech to text model.

    .. todo::

        Document this

    Args:
        encoder:
        decoder:
    """

    def __init__(
        self,
        alphabet: Alphabet,
        model: EncoderDecoder,
        loss: torch.nn.CTCLoss,
        pre_process_steps: List[Tuple[Callable, bool]],
        post_process: Union[None, CTCGreedyDecoder, CTCBeamDecoder],
    ):
        super().__init__()
        self.alphabet = alphabet
        self.model = model
        self.loss = loss
        self.pre_process_steps = pre_process_steps
        self.post_process = post_process
