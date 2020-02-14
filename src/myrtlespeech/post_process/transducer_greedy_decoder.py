from typing import List
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNNState
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

    def decode(
        self,
        inp: Tuple[torch.Tensor, torch.Tensor],
        hx_enc: Optional[RNNState] = None,
        hx_pred: Optional[RNNState] = None,
    ) -> Tuple[List[torch.tensor], Tuple[RNNState, RNNState]]:
        """Greedy Transducer decode method.

        See :py:class:`TransducerDecoderBase` for args.
        """

        (fs, fs_lens), hx_enc = self._model.encode(inp, hx_enc)
        fs = fs[
            : fs_lens.max(), :, :
        ]  # size: seq_len, batch = 1, rnn_features

        hx_pred = None
        label: List[torch.tensor] = []

        for t in range(fs.shape[0]):

            f = fs[t, :, :].unsqueeze(0)

            # add length
            f = (f, torch.IntTensor([1]))

            not_blank = True
            symbols_added = 0
            while not_blank and symbols_added < self._max_symbols_per_step:
                g, hidden_prime = self._pred_step(
                    self._get_last_idx(label), hx_pred
                )
                logp = self._joint_step(f, g)

                # get index k, of max prob
                idx = logp.max(0).indices
                if idx == self._blank_index:
                    not_blank = False
                else:
                    label.append(idx)
                    hx_pred = hidden_prime
                symbols_added += 1

        hx_pred = self.set_hx_zeros_if_none(hx_pred, hidden_prime)

        del f, g, hidden_prime, logp, fs, fs_lens
        return label, (hx_enc, hx_pred)  # type: ignore
