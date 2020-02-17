from typing import List
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNNState
from myrtlespeech.model.transducer import Transducer


class TransducerDecoderBase(torch.nn.Module):
    r"""Base Transducer decoder class.

    .. note::

        This class should not be instantiated directly. Instead use specific
        decoders (e.g. :py:class:`TransducerGreedyDecoder` or
        :py:class:`TransducerBeamDecoder`).


    Args:
        blank_index: Index of the "blank" symbol. It is advised that the blank
            symbol is placed at the end of the alphabet in order to avoid
            different symbol index conventions in the prediction and joint
            networks (i.e. input and output of Transducer) but this condition
            is not enforced.

        model: An :py:class:`myrtlespeech.model.transducer.Transducer` model
            to use during decoding. See the
            :py:class:`myrtlespeech.model.transducer.Transducer`
            docstring for more information.

        max_symbols_per_step: The maximum number of symbols that can be added
            to the output sequence in a single time step.
    """

    def __init__(
        self,
        blank_index: int,
        model: Transducer,
        max_symbols_per_step: int = 100,
    ):
        if blank_index < 0:
            raise ValueError(f"blank_index={blank_index} must be >= 0")

        assert max_symbols_per_step >= 0, "max_symbols_per_step must be >= 0"

        assert isinstance(
            model, Transducer
        ), "To perform Transducer decoding, model must be a Transducer"

        super().__init__()
        self._blank_index = torch.tensor([blank_index])
        self._model = model
        self._max_symbols_per_step = max_symbols_per_step
        self._SOS = torch.tensor([-10])  # Start of sequence
        self._device = "cuda:0" if self._model.use_cuda else "cpu"

    @torch.no_grad()
    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hxs: Optional[Tuple[RNNState, RNNState]] = None,
    ) -> Tuple[torch.Tensor, Tuple[RNNState, RNNState]]:
        """Decodes transducer outputs returning :py:class:`torch.Tensors`."""
        preds, hid_tensors = self.forward_list(x, hxs)
        return self._collate_preds(preds), hid_tensors

    @torch.no_grad()
    def forward_list(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hxs: Optional[Tuple[RNNState, RNNState]] = None,
    ) -> Tuple[List[List[torch.Tensor]], Tuple[RNNState, RNNState]]:
        r"""Decodes Transducer output.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple of inputs to the network where both elements are
                :py:class:`torch.Tensor`s. ``x[0]`` is the audio
                feature input with size ``[batch, channels, features,
                max_input_seq_len]`` while ``x[1]`` is the audio input lengths
                of size ``[batch]``.

            hxs: An Optional List of length ``batch`` where each element is a
                Tuple of encoder and prediction network input ``RNNState``s.

        Returns:
            A Tuple of the form ``(a, b)`` where ``a`` is a List of Lists
            where each sublist contains the index predictions of the decoder
            and ``b`` is a List of length ``batch`` where each element is a
            Tuple of encoder and prediction network output ``RNNState``s.
        """
        training_state = self._model.training
        self._model.eval()

        preds = []
        hids = []
        hid_inp0: Optional[RNNState] = None
        hid_inp1: Optional[RNNState] = None
        for b in range(x[0].shape[0]):
            audio_len = x[1][b].unsqueeze(0)
            audio_features = x[0][b, :, :, :audio_len].unsqueeze(0)
            audio_inp = (audio_features, audio_len)
            if hxs is not None:
                hid_inp0 = self._get_hidden_state(hxs[0], batch=b)
                hid_inp1 = self._get_hidden_state(hxs[1], batch=b)
            sentence, hid = self.decode(audio_inp, hid_inp0, hid_inp1)
            preds.append(sentence)
            hids.append(hid)

        # restore training state
        self._model.train(training_state)

        hid_tensors = self._collate_hid_states(hids)

        del audio_inp, audio_features, audio_len
        return preds, hid_tensors

    def _collate_preds(self, preds: List[torch.Tensor]) -> torch.Tensor:
        """Collates list of predictions to batched form."""
        # use negative padding_value (i.e. an index that isn't valid)
        padding_value = -1
        sequences: List[torch.Tensor] = []
        for pred in preds:
            pred = [p.view((-1)) for p in pred]
            if pred == []:
                pred = torch.tensor([])
            else:
                pred = torch.cat(pred)
            sequences.append(pred.type(torch.int32))

        max_size = sequences[0].size()
        leading_dims = max_size[:-1]
        max_len = max([s.size(-1) for s in sequences])

        out_dims = (len(sequences),) + leading_dims + (max_len,)

        out_tensor = (torch.ones(*out_dims) * padding_value).type(torch.int32)
        for i, tensor in enumerate(sequences):
            length = tensor.size(-1)
            out_tensor[i, ..., :length] = tensor
        return out_tensor

    def _get_hidden_state(self, hx: RNNState, batch: int) -> RNNState:
        """Returns ``RNNState`` of ``batch`` index."""
        if isinstance(hx, tuple):
            hx_inp0 = hx[0][:, batch, :].unsqueeze(1)
            hx_inp1 = hx[1][:, batch, :].unsqueeze(1)
            hx_inp = (hx_inp0, hx_inp1)
        else:
            hx_inp = hx[:, batch, :].unsqueeze(1)
        return hx_inp

    def _collate_hid_states(
        self, states: List[Tuple[RNNState, RNNState]]
    ) -> Tuple[RNNState, RNNState]:
        """Collates List of ``RNNState``s into :py:class`torch.Tensor` form."""
        h0s: List = []
        h1s: List = []
        h00s, h01s, h10s, h11s = [], [], [], []
        for h0, h1 in states:
            if isinstance(h0, tuple):
                h00, h01 = h0
                h10, h11 = h1
                h00s.append(h00)
                h01s.append(h01)
                h10s.append(h10)
                h11s.append(h11)
            else:
                h0s.append(h0)
                h1s.append(h1)
        if not h0s:
            h0 = torch.cat(h00s, dim=1), torch.cat(h01s, dim=1)
            h1 = torch.cat(h10s, dim=1), torch.cat(h11s, dim=1)
        else:
            h0 = torch.cat(h0s, dim=1)
            h1 = torch.cat(h1s, dim=1)
        return h0, h1

    def decode(
        self,
        inp: Tuple[torch.Tensor, torch.Tensor],
        hx_enc: Optional[RNNState] = None,
        hx_pred: Optional[RNNState] = None,
    ) -> Tuple[List[int], Tuple[RNNState, RNNState]]:
        r"""Decodes a single sample.

        Args:
            inp: Tuple where the first element is the transducer encoder
                input (a :py:class:`torch.Tensor`) with size ``[batch,
                channels, features, max_input_seq_len]`` and the second element
                is a :py:class:`torch.Tensor` of size ``[batch]`` where each
                entry represents the sequence length of the corresponding
                *input* sequence to the transducer encoder.

            hx_enc: Optional ``RNNState`` of the encoder.

            hx_pred: Optional ``RNNState`` of the prediction network.

        Returns:
            A Tuple of the form ``(a, (b, c))`` where ``a`` is a List of
            length ``[batch]`` where each element is a List of indexes. ``b``
            and ``c`` are the ``RNNState`` of the encoder and prediction
            networks respectively.
        """
        raise NotImplementedError(
            "decode method not implemented. Do not \
            instantiate `TransducerDecoderBase` directly. Instead use a \
            specific e.g. `TransducerGreedyDecoder`"
        )

    def _pred_step(
        self, label: torch.Tensor, hidden: RNNState
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RNNState]:
        r"""Performs a step of the transducer prediction network."""
        if label == self._SOS:
            y = None
        else:
            if label > self._blank_index:
                label -= 1  # Since ``output indices = input indices + 1``
                # when ``index > self._blank_index``.
            y = label.view((1, 1)), torch.IntTensor([1])
            y = y[0].to(self._device), y[1].to(self._device)
        return self._model.predict_net.predict(y, hidden, decoding=True)

    def _joint_step(
        self,
        enc: Tuple[torch.Tensor, torch.Tensor],
        pred: Tuple[torch.Tensor, torch.Tensor],
    ):
        r"""Performs a step of the transducer joint network."""

        logits, _ = self._model.joint_net((enc, pred))
        res = torch.nn.functional.log_softmax(logits, dim=-1).squeeze()
        assert (
            len(res.shape) == 1
        ), "this _joint_step result should have just one non-zero dimension"

        return res

    def _get_last_idx(self, labels: List[torch.tensor]) -> torch.tensor:
        r"""Returns the final index in a list of indexes."""
        idx = self._SOS if labels == [] else labels[-1]
        if isinstance(idx, int):
            idx = torch.IntTensor([idx]).view((1))
        return idx

    def set_hx_zeros_if_none(
        self, hx_orig: Optional[RNNState], hx_like: RNNState
    ) -> RNNState:
        """Returns zero state like ``hx``.

        This is used in case in which decoder outputs all blanks and hence
        the hidden state is None."""
        # Set hidden state to zero vector for case in which all
        # blanks are output.
        if hx_orig is None:
            if isinstance(hx_like, tuple):
                hn, cn = hx_like
                hx_orig = torch.zeros_like(hn), torch.zeros_like(cn)
            else:
                hx_orig = torch.zeros_like(hx_like)
        return hx_orig

    def __repr__(self):
        str = self._get_name() + "("
        str += f"max_symbols_per_step={self._max_symbols_per_step}, "
        str += f"blank_index={self._blank_index}"
        return str + ")"
