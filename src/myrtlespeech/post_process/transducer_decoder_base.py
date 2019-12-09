from typing import List
from typing import Tuple

import torch
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

        assert (
            max_symbols_per_step > 0
        ), "max_symbols_per_step must be a positive integer"

        assert isinstance(
            model, Transducer
        ), "To perform Transducer decoding, model must be a Transducer"

        super().__init__()
        self._blank_index = blank_index
        self._model = model
        self._max_symbols_per_step = max_symbols_per_step
        self._SOS = -1  # Start of sequence
        self._device = "cuda:0" if self._model.use_cuda else "cpu"

    @torch.no_grad()
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> List[List[int]]:
        r"""Decodes Transducer output.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple of inputs to the network where both elements are
                :py:class:`torch.Tensor`s. ``x[0]`` is the audio
                feature input with size ``[batch, channels, features,
                max_input_seq_len]`` while ``x[0]`` is the audio input lengths
                of size ``[batch]``.

        Returns:
            A List of Lists where each sublist contains the index predictions
            of the decoder.
        """
        training_state = self._model.training
        self._model.eval()

        preds = []
        for b in range(x[0].shape[0]):
            audio_len = x[1][b].unsqueeze(0)
            audio_features = x[0][b, :, :, :audio_len].unsqueeze(0)
            audio_inp = (audio_features, audio_len)
            sentence = self.decode(audio_inp)
            preds.append(sentence)

        # restore training state
        self._model.train(training_state)

        del audio_inp, audio_features, audio_len
        return preds

    def decode(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:
        r"""Decodes a single sample.

        Args:
            inp: Tuple where the first element is the transducer encoder
                input (a :py:class:`torch.Tensor`) with size ``[batch,
                channels, features, max_input_seq_len]`` and the second element
                is a :py:class:`torch.Tensor` of size ``[batch]`` where each
                entry represents the sequence length of the corresponding
                *input* sequence to the transducer encoder.

        Returns:
            A List of length ``[batch]`` where each element is a List of
            indexes.

        """
        raise NotImplementedError(
            "decode method not implemented. Do not \
            instantiate `TransducerDecoderBase` directly. Instead use a \
            specific e.g. `TransducerGreedyDecoder`"
        )

    def _pred_step(self, label, hidden):
        r"""Performs a step of the transducer prediction network."""
        if label == self._SOS:
            y = None
        else:
            if label > self._blank_index:
                label -= 1  # Since ``output indices = input indices + 1``
                # when ``index > self._blank_index``.
            y = torch.IntTensor([[label]]), torch.IntTensor([1])
            y = y[0].to(self._device), y[1].to(self._device)
        (out, hid), lengths = self._model.predict_net.predict(
            y, hidden, decoding=True
        )
        return (out, lengths), hid

    def _joint_step(self, enc, pred):
        r"""Performs a step of the transducer joint network."""

        logits, _ = self._model.joint_net((enc, pred))
        res = torch.nn.functional.log_softmax(logits, dim=-1).squeeze()
        assert (
            len(res.shape) == 1
        ), "this _joint_step result should have just one non-zero dimension"

        return res

    def _get_last_idx(self, labels):
        r"""Returns the final index in a list of indexes."""
        return self._SOS if labels == [] else labels[-1]

    def __repr__(self):
        str = self._get_name() + "("
        str += f"max_symbols_per_step={self._max_symbols_per_step}, "
        str += f"blank_index={self._blank_index}"
        return str + ")"
