from typing import Callable
from typing import Optional
from typing import Tuple

import torch


class RNNTEncoder(torch.nn.Module):
    r"""`RNN-T <https://arxiv.org/pdf/1211.3711.pdf>`_ encoder. Architecture
    based mainly on `Streaming End-to-end Speech Recognition For Mobile Devices
    <https://arxiv.org/pdf/1811.06621.pdf>`_.

    Args:

        rnn1: A :py:class:`torch.nn.Module` containing the first recurrent part of
            the RNN-T encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py:`torch.Tensor`) with size ``[max_cnn_seq_len, batch,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size
            ``[max_rnn_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling.

        time_reducer: An Optional ``Callable`` that reduces the number of timesteps
            into ``rnn2`` by stacking adjacent frames in the frequency dimension as
            employed in https://arxiv.org/pdf/1811.06621.pdf. This function takes two arguments:
            a Tuple of ``rnn1`` output and the sequence lengths of size ``[batch]``,
            and an optional ``time_reduction_factor`` (see below).
            This callable must return a Tuple where the first element is the result after
            applying the module to the input which must have size
            ``[ceil(max_rnn_seq_len/time_reduction_factor) , batch, rnn_features * time_reduction_factor ]``.
            The second element of the tuple must be a :py:class:`torch.Tensor` of size ``[batch]``
            that contains the new length of the corresponding sequence.

        time_reduction_factor: An Optional ``int`` with default value 2. If
            ``time_reducer`` is not None, this is the ratio by which the time dimension
            is reduced.

        rnn2: An Optional :py:class:`torch.nn.Module` containing the second
            recurrent part of the RNN-T encoder. This must be None unless ``time_reducer``
            is also passed.

            Must accept as input a tuple where the first element is the output
            from time_reducer (a :py:`torch.Tensor`) with size ``[max_downsampled_seq_len, batch,
            rnn1_features * time_reduction_factor]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the cnn output. It must have size
            ``[max_downsampled_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling.

    Returns:
        The output of ``rnn2`` if it is not None, else the output of ``rnn1``.
        The first element of the tensor output is transposed so that it is
        of size ``[batch, new_seq_len, out_features]``

    """

    def __init__(
        self,
        rnn1: torch.nn.Module,
        time_reducer: Optional[Callable] = None,
        time_reduction_factor: Optional[int] = 2,
        rnn2: Optional[torch.nn.Module] = None,
    ):
        if time_reducer is None:
            assert (
                rnn2 is None
            ), "Do not pass rnn2 without a time_reducer Callable"
        else:
            assert isinstance(time_reduction_factor, int)
            assert (
                time_reduction_factor > 1
            ), f"time_reduction_factor must be > 2 but = {time_reduction_factor}"

        assert rnn1.batch_first == False
        if rnn2:
            assert rnn2.batch_first == False

        super().__init__()

        self.rnn1 = rnn1
        self.time_reducer = time_reducer
        self.time_reduction_factor = time_reduction_factor
        self.rnn2 = rnn2

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.rnn1 = self.rnn1.cuda()
            if self.rnn2 is not None:
                self.rnn2 = self.rnn2.cuda()

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the result of applying the RNN-T encoder to the input audio feautures.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        See :py:class:`.RNNTEncoder` for detailed information about the input
        and output of each module.

        Args:
            x: Input for the first rnn module. See initialisation docstring.

        Returns:
            Output from ``rnn2`` if present, else output from ``rnn1``. See initialisation
            docstring.
        """

        if self.use_cuda:
            h = (x[0].cuda(), x[1].cuda())

        h = self.rnn1(h)

        if self.time_reducer:
            h = self.time_reducer(h, self.time_reduction_factor)

            h = self.rnn2(h)

        return (h[0].transpose(0, 1).contiguous(), h[1])
