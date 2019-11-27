from typing import Optional
from typing import Tuple

import torch


class Transducer(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ Network.

    Args:
        encoder: A :py:class:`torch.nn.Module` to use as the encoder. Note
            that this network is also referred to as the Transcription Network
            in the literature. The `encoder` must accept as input a tuple
            where the first element is the network input: a
            :py:`torch.Tensor`) with size ``[batch, channels,
            features, max_input_seq_len]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size
            ``[max_out_seq_len, batch, output_features]``. The second element
            of the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling in `encoder`.

            It is possible *but not necessary* to use an initialized
            :py:class:`TransducerEncoder` class as the `encoder`.

        embedding: A :py:class:`torch.nn.Module` which is an embedding lookup
            for targets (eg graphemes, wordpieces) which accepts a
            :py:`torch.Tensor` as input of size
            ``[batch, max_output_seq_len]``.

        dec_rnn: A :py:class:`torch.nn.Module` containing the recurrent part of
            the Transducer prediction.

            Must accept as input a tuple where the first element is the
            prediction network input (a :py:`torch.Tensor`) with size ``[batch,
            max_output_seq_len + 1, in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represent the sequence length of the corresponding *input*
            sequence to the rnn. *Note: this must be a `batch_first` rnn.*

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size ``[batch,
            max_output_seq_len + 1, in_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence.

        fully_connected: A :py:class:`torch.nn.Module` containing the fully
            connected part of the Transducer joint network.

            Must accept as input a tuple where the first element is
            a :py:class:`torch.Tensor` with size ``[batch,
            encoder_out_seq_len, dec_rnn_out_seq_len, hidden_dim]`` where
            `hidden_dim` is the the concatenation of the `encoder` and
            `dec_rnn` outputs along the hidden dimension axis. The second
            element is and the input (a :py:class:`torch.Tensor`) with size
            ``[batch, max_fc_in_seq_len, max_fc_in_features]`` and the second
            element is a :py:class:`torch.Tensor` of size ``[batch]``
            where each entry represents the sequence length of the
            corresponding *input* sequence to the fully connected layer(s).

            It must return a tuple where the first element is the result after
            applying the module to the previous layers output. It must have
            size ``[batch, max_out_seq_len, out_features]``. The second element
            is returned unchanged but in this context should be a
            :py:class:`torch.Tensor` of size ``[batch]`` that contains the
            *output* lengths of the audio features inputs.

    """

    def __init__(self, encoder, embedding, dec_rnn, fully_connected):
        super().__init__()

        assert (
            dec_rnn.batch_first is True
        ), "dec_rnn should be a batch_first rnn"

        self.encode = encoder

        self.predict_net = self._predict_net(embedding, dec_rnn)

        self.joint_net = self._joint_net(fully_connected)

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.cuda()

    @staticmethod
    def _predict_net(embedding, dec_rnn):
        return torch.nn.ModuleDict({"embed": embedding, "dec_rnn": dec_rnn})

    @staticmethod
    def _joint_net(fully_connected):
        return torch.nn.ModuleDict({"fully_connected": fully_connected})

    def forward(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the Transducer network to `x`.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple ``(x[0], x[1])``. ``x[0]`` is input to the network and
                is a Tuple ``x[0] = (x[0][0], x[0][1])`` where both elements
                are :py:class:`torch.Tensor`s. ``x[0][0]`` is the audio feature
                input with  size ``[batch, channels, features,
                max_input_seq_len]`` while ``x[0][1]`` is the target label
                tensor of size ``[batch, max_label_length]``.

                ``x[1]`` is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *input* lengths of a) the
                audio feature inputs ``x[1][0]`` and b) the target sequences
                ``x[1][1]``.

        Returns:
            A Tuple where the first element is the unnormalized output of the
            :py:class`Transducer` network: a :py:class:`torch.Tensor` with
            size ``[batch, max_seq_len, max_label_length + 1,
            vocab_size + 1]``. `max_seq_len` is the
            length of the longest sequence in the batch output from the
            Encoder network whereas `max_label_seq_len` is the length of
            the longest *label* sequence in the batch output from the
            :py:class:`Prediction` network. Note that the dimension at
            index 2 is `max_label_seq_len + 1` since the
            start-of-sequence label is prepended to the label sequence and
            the dimension at index 3 is `vocab_size + 1` because the blank
            symbol can be optput.

            The second element is a :py:class:`torch.Tensor` of size
            ``[batch]`` that contains the Encoder sequence output lengths.
            """

        self._certify_inputs_forward(x)

        audio_data, label_data = self._prepare_inputs_forward(x, self.use_cuda)

        f = self.encode(audio_data)  # f[0] = (T, B, H1)
        g = self.prediction(label_data)  # g[0] = (U, B, H2)

        joint_inp = self._enc_pred_to_joint(f, g)
        out = self.joint(joint_inp)

        del f, g, audio_data, label_data, x, joint_inp

        return out

    def prediction(
        self, y: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the Transducer prediction network.

        This function is only appropriate during training when the
        ground-truth labels are available.

        .. note:: The length of the sequence is increased by one as the
        start-of-sequence embedded state (all zeros) is prepended to the start
        of the label sequence. Note that this change *is not* reflected in the
        output lengths as the :py:class:`TransducerLoss` requires the true
        label lengths.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            y: A Tuple where the first element is the target label tensor of
                size ``[batch, max_label_length]`` and the second is a
                :py:class:`torch.Tensor` of size ``[batch]`` that contains the
                *input* lengths of these target label sequences.

        Returns:
            Output from ``dec_rnn``. See initialisation docstring.
        """

        y = self.embedding(y)
        y = self._prepend_SOS(y)
        # Update the lengths with +1 for input to the dec_rnn
        y = (y[0], y[1] + 1)
        out = self.dec_rnn(y)
        # Revert the lengths to 'true' values (i.e. not including SOS)
        out = (out[0], out[1] - 1)

        del y

        return out

    def embedding(
        self, y: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Wrapper function on `self._embedding`.

        Casts inputs to int64 if necessary before applying embedding.

        Args:
            y: A Tuple where the first element is the target label tensor of
                size ``[batch, max_label_length]`` and the second is a
                :py:class:`torch.Tensor` of size ``[batch]`` that contains the
                *input* lengths of these target label sequences.

        Returns:
            A Tuple where the first element is the embedded label tensor of
            size ``[batch, max_label_length, hidden_size]`` and the second
            is a :py:class:`torch.Tensor` of size ``[batch]`` that contains
            the *output* lengths of these target label sequences. These
            lengths will be the same as the input lengths.
        """

        y_0, y_1 = y

        if y_0.dtype != torch.long:
            y_0 = y_0.long()

        out = (self.predict_net["embed"](y_0), y_1)

        del y, y_0, y_1

        return out

    @staticmethod
    def _prepend_SOS(y):
        r"""Prepends the SOS embedding (all zeros) to the target tensor.
        """
        y_0, y_1 = y

        B, _, H = y_0.shape
        # preprend blank
        start = torch.zeros((B, 1, H)).type(y_0.dtype).to(y_0.device)
        y_0 = torch.cat([start, y_0], dim=1)  # (B, U + 1, H)
        y_0 = y_0.contiguous()

        del start, y

        return (y_0, y_1)

    def joint(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the Transducer joint network.

        Args:
            x: A Tuple ``(x[0], x[1])``. ``x[0][0]`` is the first element of
                the `encoder` output (i.e. `.encode(...)[0]` - see
                :py:class:`.TransducerEncoder` docstring). ``x[0][1]`` is the
                first element of the prediction network output (i.e.
                `.prediction(...)[0]` - see prediction docstring).

                ``x[1]`` is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *input* lengths of a) the
                audio feature inputs ``x[1][0]`` and b) the target sequences
                ``x[1][1]``.

        Returns:
            The output of the :py:class:`.Transducer` network. See
            :py:class:`.Transducer`'s :py:meth:`forward` docstring.
        """
        (f, g), seq_lengths = x

        T, B1, H1 = f.shape
        B2, U_, H2 = g.shape

        assert (
            B1 == B2
        ), "Batch size from prediction network and transcription must be equal"

        f = f.transpose(1, 0)  # (T, B, H1) -> (B, T, H1)
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        f = f.expand((B1, T, U_, H1)).contiguous()

        g = g.unsqueeze(dim=1)  # (B, 1, U_, H)
        g = g.expand((B1, T, U_, H2)).contiguous()

        joint_inp = torch.cat(
            [f, g], dim=3
        ).contiguous()  # (B, T, U_, H1 + H2)

        del g, f, x

        # drop label_seq_lens (see :py:class:`Transducer` forward docstring)
        h = self.joint_net["fully_connected"]((joint_inp, seq_lengths[0]))

        del joint_inp

        return h

    @staticmethod
    def _certify_inputs_forward(inp):
        try:
            ((x, y), (x_lens, y_lens)) = inp
        except ValueError:
            raise ValueError(
                "Unable to unpack inputs to Transducer. Are you using the \
            `TransducerTraining()` callback \
            found in `myrtlespeech.run.callbacks.rnn_t_training`?"
            )
        B1, C, I, T = x.shape
        B2, U = y.shape

        (B3,) = x_lens.shape
        (B4,) = y_lens.shape
        assert (
            B1 == B2 == B3 == B4
        ), "Batch size must be the same for inputs and targets"

        if not (x_lens <= T).all():
            raise ValueError(
                "x_lens must be less than or equal to max number of time-steps"
            )
        if not (y_lens <= U).all():
            raise ValueError(
                f"y_lens must be less than or equal to max number of output \
                symbols but {y_lens.max()} > {U}"
            )

        return (B1, C, I, T, U)  # return dimensions

    @staticmethod
    def _prepare_inputs_forward(inp, use_cuda):
        ((x_inp, y), (x_lens, y_lens)) = inp
        if use_cuda:
            x_inp = x_inp.cuda()
            x_lens = x_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()

        audio_data = (x_inp, x_lens)
        label_data = (y, y_lens)

        del x_inp, y, x_lens, y_lens, inp

        return audio_data, label_data

    @staticmethod
    def _enc_pred_to_joint(f, g):
        f_inp = f[0]
        g_inp = g[0]
        seq_lengths = (f[1], g[1])  # (time_lens, label_lens)
        del f, g
        return ((f_inp, g_inp), seq_lengths)

    @property
    def dec_rnn(self):
        r"""Decoder RNN (in prediction network)."""
        return self.predict_net["dec_rnn"]


class TransducerEncoder(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ encoder.

    Alternatively referred to as Transducer transcription network. All of the
    submodules other than ``rnn1` are Optional.

    .. note:: If present, the modules are applied in the following order:
        ``fc1`` -> ``rnn1`` -> ``fc2``

    Args:
        rnn1: A :py:class:`torch.nn.Module` containing the first recurrent part
            of the Transducer encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py:`torch.Tensor`) with size ``[max_seq_len, batch,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size
            ``[max_rnn_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence.

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
            input (a :py`torch.Tensor`) with size ``[max_downsampled_seq_len,
            batch, rnn_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the fc layer.

            It must return a tuple where the first element is the result after
            applying this fc layer over the final dimension only. This layer
            *must not change* the hidden size dimension so this has size
            ``[max_downsampled_seq_len, batch, rnn_features]``. The second
            element of the tuple return value is a :py:class:`torch.Tensor`
            with size ``[batch]`` where each entry represents the sequence
            length of the corresponding *output* sequence.


    Returns:
        A Tuple where the first element is the  output of ``fc2`` if it is not
        None, else the output of `rnn1` and the second element is a
        :py:class:`torch.Tensor` of size ``[batch]`` where each entry
        represents the sequence length of the corresponding *output*
        sequence to the encoder.
    """

    def __init__(
        self,
        rnn1: torch.nn.Module,
        fc1: Optional[torch.nn.Module] = None,
        fc2: Optional[torch.nn.Module] = None,
    ):

        assert rnn1.rnn.batch_first is False

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

        # add extra hard tahn
        tanh = torch.nn.Hardtanh(min_val=0.0, max_val=20.0)
        self.hardtanh = lambda x: (tanh(x[0]), x[1])

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns result of applying the encoder to the audio features.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        See :py:class:`.TransducerEncoder` for detailed information about the
        input and output of each module.

        Args:
            x: Tuple where the first element is the encoder
                input (a :py:`torch.Tensor`) with size ``[batch, channels,
                features, max_input_seq_len]`` and the second element is a
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence to the rnn. The channel dimension contains context
                frames and is immediately flattened into the `features`
                dimension. This reshaping operation is not dealt with in
                preprocessing so that: a) this model conforms to existing
                pre-processing pipeline, and b) because future edits to
                `myrtlespeech.model.rnn_t.TransducerEncoder` may add
                convolutions before input to the first layer `fc1`/`rnn1`.

        Returns:
            Output from `fc2`` if present else output from ``rnn1``. See
            initialisation docstring.
        """

        self._certify_inputs_encode(x)

        if self.use_cuda:
            h = (x[0].cuda(), x[1].cuda())
        else:
            h = x

        # Add Optional convolutions here in the future?
        h = self._prepare_inputs_fc1(h)
        if hasattr(self, "fc1"):
            h = h[0].transpose(2, 3), h[1]
            h = self.fc1(h)
            h = h[0].transpose(2, 3), h[1]
            h = self.hardtanh(h)
        h = self._prepare_inputs_rnn1(h)

        h = self.rnn1(h)

        if hasattr(self, "fc2"):
            h = self.fc2(h)
            h = self.hardtanh(h)
        del x

        return h

    @staticmethod
    def _certify_inputs_encode(inp):

        (x, x_lens) = inp
        B1, C, I, T = x.shape
        (B2,) = x_lens.shape
        assert B1 == B2, "Batch size must be the same for inputs and targets"
        del x, x_lens, inp

    @staticmethod
    def _prepare_inputs_rnn1(inp):
        r"""Reshapes inputs to prepare for `rnn1`.
        """
        (x, x_lens) = inp
        B, C, I, T = x.shape

        assert C == 1, f"There should only be a single channel input but C={C}"

        x = x.squeeze(1)  # B, I, T
        x = x.permute(2, 0, 1).contiguous()  # T, B, I
        del inp
        return (x, x_lens)

    @staticmethod
    def _prepare_inputs_fc1(inp):
        r"""Reshapes inputs to prepare them for `fc1`.

        This involves flattening n_context in channel dimension in the hidden
        dimension.
        """

        (x, x_lens) = inp
        B, C, I, T = x.shape

        if not C == 1:
            x = x.view(B, 1, C * I, T).contiguous()
        del inp
        return (x, x_lens)
