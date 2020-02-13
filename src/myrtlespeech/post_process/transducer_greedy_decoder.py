from typing import List
from typing import Tuple

import torch
from myrtlespeech.model.transducer import Transducer
from myrtlespeech.post_process.transducer_decoder_base import (
    TransducerDecoderBase,
)


class TransducerGreedyDecoder(TransducerDecoderBase):
    """Decodes Transducer output using a greedy strategy.

    See :py:class:`TransducerDecoderBase` for args.
    """

    def __init__(
        self,
        blank_index: int,
        model: Transducer,
        max_symbols_per_step: int = 100,
    ):

        super().__init__(
            blank_index=blank_index,
            model=model,
            max_symbols_per_step=max_symbols_per_step,
        )

    def decode(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:
        """Greedy Transducer decode method.

        See :py:class:`TransducerDecoderBase` for args.
        """

        (fs, fs_lens), _ = self._model.encode(inp)
        fs = fs[
            : fs_lens.max(), :, :
        ]  # size: seq_len, batch = 1, rnn_features

        hidden = None
        label: List[int] = []

        for t in range(fs.shape[0]):

            f = fs[t, :, :].unsqueeze(0)

            # add length
            f = (f, torch.IntTensor([1]))

            not_blank = True
            symbols_added = 0
            while not_blank and symbols_added < self._max_symbols_per_step:

                g, hidden_prime = self._pred_step(
                    self._get_last_idx(label), hidden
                )
                logp = self._joint_step(f, g)

                # get index k, of max prob
                max_val, idx = logp.max(0)
                idx = idx.item()

                if idx == self._blank_index:
                    not_blank = False
                else:
                    label.append(idx)
                    hidden = hidden_prime
                symbols_added += 1

        del f, g, hidden, hidden_prime, logp, fs, fs_lens
        return label
