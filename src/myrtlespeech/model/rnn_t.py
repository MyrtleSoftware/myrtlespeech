from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNNState


class RNNTEncoder(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ encoder.

    Alternatively referred to as Transducer transcription network. All of the
    submodules other than ``rnn1`` are Optional.

    .. note::

        If present, the modules are applied in the following order:
        ``fc1 -> rnn1 -> fc2``

    Args:
        rnn1: A :py:class:`torch.nn.Module` containing the first recurrent part
            of the Transducer encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py:class:`torch.Tensor`) with size ``[max_seq_len, batch,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a nested Tuple of the form ``((a, b), c)``
            where ``a`` is the result after applying the module to the input:
            a :py:class:`torch.Tensor` of size ``[max_rnn_seq_len, batch,
            rnn_features]``. ``b`` is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. ``c`` is the ``RNNState`` of the
            module.

        fc1: An Optional :py:class:`torch.nn.Module` containing the first fully
            connected part of the Transducer encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py`torch.Tensor`) with size ``[max_seq_len, batch,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the fc layer.

            It must return a tuple where the first element is the result after
            applying this fc layer over the final dimension only. This layer
            *must not change* the hidden size dimension so this has size
            ``[max_seq_len, batch, in_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence (which will be identical to the
            input lengths).

        fc2: An Optional :py:class:`torch.nn.Module` containing the second
            fully connected part of the Transducer encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py`torch.Tensor`) with size ``[max_seq_len,
            batch, rnn_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the fc layer.

            It must return a tuple where the first element is the result after
            applying this fc layer over the final dimension only. This layer
            *must not change* the hidden size dimension so this has size
            ``[max_seq_len, batch, rnn_features]``. The second
            element of the tuple return value is a :py:class:`torch.Tensor`
            with size ``[batch]`` where each entry represents the sequence
            length of the corresponding *output* sequence.


    Returns:
        A Tuple where the first element is the  output of ``fc2`` if present
        None, else the output of ``rnn1`` and the second element is the
        ``RNNState`` of ``rnn1``.
    """

    def __init__(
        self,
        rnn1: torch.nn.Module,
        fc1: Optional[torch.nn.Module] = None,
        fc2: Optional[torch.nn.Module] = None,
    ):

        super().__init__()

        if fc1:
            self.fc1 = fc1
        self.rnn1 = rnn1
        if fc2:
            self.fc2 = fc2

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            if fc1 is not None:
                self.fc1 = self.fc1.cuda()
            self.rnn1 = self.rnn1.cuda()
            if fc2 is not None:
                self.fc2 = self.fc2.cuda()

        hardtanh = torch.nn.Hardtanh(min_val=0.0, max_val=20.0)
        self.hardtanh = lambda x: (hardtanh(x[0]), x[1])

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hx: Optional[RNNState] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RNNState]:
        r"""Returns result of applying the encoder to the audio features.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple where the first element is the audio feature sequence:
                a :py:class:`torch.Tensor` with size ``[batch, channels,
                features, max_input_seq_len]`` and the second element is a
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding audio
                feature input. The channel dimension contains
                context frames and is immediately flattened into the
                ``features`` dimension. This reshaping operation is not dealt
                with in preprocessing so that: a) this model conforms to the
                existing pre-processing pipeline, and b) because future edits
                to this class may add convolutions before input to the first
                layer ``fc1``/``rnn1``.

            hx: An Optional ``RNNState`` of ``rnn1``.

        Returns:
            See initialisation docstring.
        """
        inp, x_lens = x
        B1, C, I, T = inp.shape
        (B2,) = x_lens.shape
        assert B1 == B2, "Batch size must be the same for inputs and targets"

        if self.use_cuda:
            h = (inp.cuda(), x_lens.cuda())
        else:
            h = x

        h = self._prepare_inputs_fc1(h)

        if hasattr(self, "fc1"):
            h = self.fc1(h)
            h = self.hardtanh(h)

        h, hid = self.rnn1(h, hx=hx)

        if hasattr(self, "fc2"):
            h = self.fc2(h)
            h = self.hardtanh(h)

        return h, hid

    @staticmethod
    def _prepare_inputs_fc1(inp):
        r"""Reshapes inputs to prepare them for ``fc1``.

        The overall transformation is: ````[batch, channels, features,
        max_input_seq_len] -> [max_input_seq_len, batch,
        channels * features]``.
        """
        x, x_lens = inp
        B, C, I, T = x.shape
        x = x.view(B, C * I, T)
        x = x.permute(2, 0, 1)  # (B, hid, T) -> (T, B, hid)
        return x.contiguous(), x_lens


class RNNTPredictNet(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ prediction network.

    Args:
        embedding: A :py:class:`torch.nn.Module` which is an embedding lookup
            for targets that must accept a :py:class:`torch.Tensor` of size
            ``[batch, max_label_len]`` as input and return a
            :py:class:`torch.Tensor` of size ``[batch, max_label_len,
            pred_nn_input_feature_size]``.

        pred_nn: A :py:class:`torch.nn.Module` containing the non-embedding
            module the Transducer prediction.

            ``pred_nn`` is a :py:class:`torch.nn.Module` with the same
            :py:meth:`forward` API as a ``batch_first``
            :py:class:`myrtlespeech.model.rnn.RNN` and the attribute
            ``hidden_size: int``.
    """

    def __init__(self, embedding: torch.nn.Module, pred_nn: torch.nn.Module):
        assert hasattr(
            pred_nn, "hidden_size"
        ), "pred_nn must have attribute `hidden_size`"
        super().__init__()
        self.embedding = embedding
        self.pred_nn = pred_nn
        self.hidden_size = pred_nn.hidden_size

    def forward(
        self,
        y: Tuple[torch.Tensor, torch.Tensor],
        hx: Optional[RNNState] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RNNState]:
        r"""Returns the result of applying the Transducer prediction network.

        .. note::

            This function is only appropriate when the
            ground-truth labels are available. The :py:meth:`predict`
            should be used for inference with ``decoding=False``.

        .. note::

            The length of the sequence is increased by ``+1`` as the
            start-of-sequence embedded state (all zeros) is prepended to the
            start of the label sequence.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            y: A Tuple where the first element is the target label tensor of
                size ``[batch, max_label_length]`` and the second is a
                :py:class:`torch.Tensor` of size ``[batch]`` that contains the
                *input* lengths of these target label sequences.

            hx: Optional ``RNNState`` of ``pred_nn``.

        Returns:
            Output from ``pred_nn``. See initialisation docstring.
        """
        return self.predict(y, hx=hx, decoding=False)

    def predict(
        self,
        y: Optional[Tuple[torch.Tensor, torch.Tensor]],
        hx: Optional[RNNState],
        decoding: bool,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RNNState]:
        r"""Excecutes :py:class:`RNNTPredictNet`.

        Args:
            y: A Optional Tuple where the first element is the target label
                tensor of size ``[batch, max_label_length]`` and the second is
                a :py:class:`torch.Tensor` of size ``[batch]`` that contains
                the *input* lengths of these target label sequences. ``y`` can
                be None iff ``decoding=True``.

            hx: The Optional ``RNNState`` of ``pred_nn``.

            decoding: A boolean. If :py:data:`True` then decoding is being
                performed and start-of-sequence token is not prepended.

        Returns:
            This will return the output of ``pred_nn`` where a hidden state is
            present iff ``decoding=True``. See :py:class:`RNN` with
            ``batch_first=True`` for API.
        """
        if not decoding:
            assert hx is None, "Do not pass hx during training"
            assert y is not None, f"y must be None during training"

        if y is None:  # then performing decoding and at start-of-sequence
            B = 1 if hx is None else hx[0].size(1)
            y = torch.zeros((1, B, self.hidden_size)), torch.IntTensor([1])
        else:
            assert (
                isinstance(y, tuple) and len(y) == 2
            ), f"y must be a tuple of length 2"
            y = self.embed(y)

        if not decoding:
            y = self._prepend_SOS(y)

        out, hid = self.pred_nn(y, hx=hx)

        return out, hid

    def embed(
        self, y: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Wrapper function on ``embedding``.

        Casts inputs to int64 if necessary before applying ``embedding``
        module.

        Args:
            y: A Tuple where the first element is the target label tensor of
                size ``[batch, max_label_length]`` and the second is a
                :py:class:`torch.Tensor` of size ``[batch]`` that contains the
                *input* lengths of these target label sequences.

        Returns:
            A Tuple where the first element is the embedded label tensor of
            size ``[batch, max_label_length, pred_nn_input_feature_size]``
            and the second is a :py:class:`torch.Tensor` of size ``[batch]``
            that contains the *output* lengths of these target label
            sequences. These lengths will be the same as the input lengths.
        """

        y_0, y_1 = y

        if y_0.dtype != torch.long:
            y_0 = y_0.long()

        out = (self.embedding(y_0), y_1)

        return out

    @staticmethod
    def _prepend_SOS(y):
        r"""Prepends the SOS embedding (all zeros) to the target tensor.
        """
        y_0, y_1 = y

        B, _, H = y_0.shape
        # preprend blank
        start = torch.zeros((B, 1, H), device=y_0.device, dtype=y_0.dtype)
        y_0 = torch.cat([start, y_0], dim=1).contiguous()  # (B, U + 1, H)

        # Update the lengths by adding one:
        return y_0, y_1 + 1


class RNNTJointNet(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ joint network.

    Args:
        fc: An :py:class:`torch.nn.Module` containing the fully connected part
            of the Transducer joint net.

            Must accept as input a tuple where the first element is the network
            input (a :py`torch.Tensor`) with size ``[batch, max_seq_len
            * max_label_len, encoder_out_feat + pred_net_out_feat]`` and the
            second element is a :py:class:`torch.Tensor` of size ``[batch]``
            where each entry represents the sequence length of the
            corresponding *input* sequence to the fc layer.

            It must return a tuple where the first element is the result after
            applying this fc layer over the final dimension only meaning it
            has size ``[batch, max_seq_len * max_label_len,
            joint_net_out_feat]``. The second element of the tuple return
            value is a :py:class:`torch.Tensor` with size ``[batch]`` where
            each entry represents the sequence length of the corresponding
            *output* sequence.
    """

    def __init__(self, fc: torch.nn.Module):
        super().__init__()
        self.fc = fc

    def forward(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the Transducer joint network.

        Args:
            x: A Tuple where the first element is the ``encoder`` output and
                the second is the ``predict_net`` output.

        Returns:
            The output of the :py:class:`.Transducer` network. See
            :py:class:`.Transducer`'s :py:meth:`forward` docstring.
        """
        (f, f_lens), (g, g_lens) = x

        T, B1, H1 = f.shape
        B2, U_, H2 = g.shape

        assert (
            B1 == B2
        ), "Batch size from prediction network and transcription must be equal"

        f = f.transpose(1, 0)  # (T, B, H1) -> (B, T, H1)
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        f = f.expand((B1, T, U_, H1))

        g = g.unsqueeze(dim=1)  # (B, 1, U_, H)
        g = g.expand((B1, T, U_, H2))

        concat_inp = torch.cat([f, g], dim=3)  # (B, T, U_, H1 + H2)

        # reshape input to give 3 dimensions instead of 4 as required by ``fc``
        concat_inp = concat_inp.view(B1, T * U_, -1)

        # drop g_lens (see :py:class:`Transducer` docstrings)
        h = self.fc((concat_inp, f_lens))

        # return to 4D shape
        h = h[0].view(B1, T, U_, -1), h[1]

        return h
