from typing import Optional, Tuple, Union

import torch

from myrtlespeech.model.utils import Lambda


class Encoder(torch.nn.Module):
    r"""

    .. todo::

        * Document this! with examples

    All ``cnn`` and ``rnn`` parameters and buffers are moved to the GPU with
    :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    """

    def __init__(
        self, cnn: Optional[torch.nn.Module], rnn: Optional[torch.nn.Module]
    ):
        super().__init__()
        self.cnn = cnn

        self.cnn_to_rnn: Optional[Lambda] = None
        if cnn is not None:
            self.cnn_to_rnn = Lambda(lambda h: conv_to_rnn_size(h))

        self.rnn = rnn

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            if self.cnn is not None:
                self.cnn = self.cnn.cuda()
            if self.rnn is not None:
                self.rnn = self.rnn.cuda()

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying ``cnn`` and ``rnn`` to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        TODO: redocument

            The value of this argument must either be
            :py:data:`None` or be a :py:class:`torch.Tensor` of size
            ``[batch]`` where each entry is an integer that gives the sequence
            length of the corresponding *input* sequence.

            When the ``seq_lens`` argument is not :py:data:`None` the encoder
            will return a tuple of ``(output, output_seq_lens)``. Here
            ``output`` is the result of applying the decoder to the input
            sequence and ``output_seq_lens`` is a :py:class:`torch.Tensor` of
            size ``[batch]`` where each entry is an integer that gives the
            sequence length of the corresponding *output* sequence.

        Args:
            x: :py:class:`torch.Tensor` or ``Tuple[torch.Tensor, torch.Tensor]``.

                At a high level, the single argument :py:class:`torch.Tensor`
                or first element of the Tuple argument must be valid input for
                the encoder. The second argument of the Tuple must contain the
                sequence length of each input.

                In more detail, for the single :py:class:`torch.Tensor`
                argument or the first argument of the Tuple:

                If ``cnn is not None`` then this must have size ``[batch,
                channels, features, max_in_seq_len]`` that is valid for
                ``cnn``. The output of ``cnn`` will be changed to have size
                ``[max_cnn_out_seq_len, batch, cnn_out_features]``. If ``rnn is
                not None`` then this must be valid for input to ``rnn``.

                If ``cnn is None and rnn is not None`` then this must have size
                ``[max_seq_len, batch, features]``.

                If ``cnn is None and rnn is None`` then the
                :py:class:`.Encoder` is the identity function that returns this
                result so size does not matter.

                For the second argument of the Tuple:

                This must have size ``[batch]`` where each entry represents the
                sequence length of the corresponding *input* sequence.

        Returns:
            The single return value or first element of the Tuple return value
            is the result after applying ``cnn`` and ``rnn`` if either, or
            both, are not None.  If ``cnn is not None or rnn is not None`` then
            it will have size ``[max_out_seq_len, batch, out_features]``.
            Otherwise it has size equal to the single argument
            :py:class:`torch.Tensor` or first element of the Tuple argument as
            the :py:class:`Encoder` acts as the identity function.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. Each of these will be less than or equal to
            ``max_out_seq_len``.
        """
        if self.use_cuda:
            x = x.cuda()
            if seq_lens is not None:
                seq_lens = seq_lens.cuda()

        h = x
        if seq_lens is not None:
            if self.cnn:
                h, seq_lens = self.cnn(h, seq_lens=seq_lens)
                assert self.cnn_to_rnn is not None
                h = self.cnn_to_rnn(h)
            if self.rnn:
                (h, _), seq_lens = self.rnn(h, seq_lens=seq_lens)
            return h, seq_lens

        if self.cnn:
            h = self.cnn(h)
            assert self.cnn_to_rnn is not None
            h = self.cnn_to_rnn(h)
        if self.rnn:
            h, _ = self.rnn(h)
        return h


def conv_to_rnn_size(x: torch.Tensor) -> torch.Tensor:
    r"""Returns a 3D :py:class:`torch.Tensor` given a 4D input.

    Args:
        x: :py:class:`torch.Tensor` with size ``[batch, channels, features,
            seq_len]``

    Returns:
        ``x`` but resized to ``[seq_len, batch, channels*features]``
    """
    batch, channels, features, seq_len = x.size()
    return x.view(batch, channels * features, seq_len).permute(2, 0, 1)
