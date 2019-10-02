from enum import Enum
from typing import Tuple

import torch


class PaddingMode(Enum):
    """Available padding modes.

    TODO
    """

    NONE = 0
    SAME = 1


def pad_same(
    length: int, kernel_size: int, stride: int = 1, dilation: int = 1
) -> Tuple[int, int]:
    r"""Returns a tuple of left and right same padding amounts.

    Consider a tensor with size ``[length]``. The aim is to pad the tensor
    such that its size after convolving it with a 1D kernel with parameters
    ``kernel_size``, ``stride``, and ``dilation`` is
    ``[ceil(length / stride)]``.

    Why? Let the indices of each element in the input tensor be:

    .. code-block:: python

        [0, 1, 2, ..., length - 1]

    For ``stride=1`` with same padding the indices of each element in the
    output tensor should be:

    .. code-block:: python

        [0, 1, 2, ..., length - 1]

    i.e. the same. When ``stride > 1`` effectively every ``stride``th element
    in this output tensor is selected, so the effective indices of the actual
    output are:

    .. code-block:: python

        [0, S, 2S, ..., stride*(ceil(length / stride) - 1)]

    This sequence has length ``[ceil(length / stride)]`` that was one of the
    requirements, great!. The index of the last element in this sequence is
    set such that it is the greatest multiple of ``stride`` that is less than
    or equal to ``length - 1``. There is no value in applying the kernel at
    an index beyond that of the last element in the input sequence. i.e. the
    following property holds:

    .. code-block:: python

        stride*(ceil(length / stride) - 1) <= length - 1

    For this element in the non-strided output to exist it must be valid to
    put the kernel at index ``stride*(ceil(length / stride) - 1)`` in the
    input sequence. The input sequence is padded to make sure this is always
    true.

    The effective kernel size with ``dilation`` is:

    .. code-block:: python

        dilation*(kernel_size - 1) + 1

    i.e. every element in the kernel except the first is expanded to
    ``dilation`` elements (the original kernel element and ``dilation - 1``
    zero elements). This results in ``dilation*(kernel_size - 1) + 1``
    elements in total.

    The total amount of padding required is, therefore:

    .. code-block:: python

        padding = (stride*(ceil(length / stride) - 1)
                + dilation*(kernel_size - 1) + 1) - 1
                - (length - 1)

    This is the index of the position of the last kernel application plus the
    number of additional element indices required at that position for it to
    be valid to apply the kernel there minus the total number element indices
    that already exist.

    This reduces to:

    .. code-block:: python

        padding = (stride*(ceil(length / stride) - 1)
                + dilation*(kernel_size - 1) + 1)
                - length

    The padding is actually split in two and concatenated to both the left
    and right side of the input tensor rather than concatenating it all to
    one side. If the total amount of padding is odd then the additional
    padding element is added to the right hand side. i.e.

    .. code-block:: python

        pad_left = padding // 2
        pad_right = padding - pad_left

        tensor = torch.cat(
            torch.zeros([pad_left]),
            tensor,
            torch.zeros([pad_right])
        )

    Args:
        length: Length of the input tensor before padding.
        kernel_size: Size of the 1D convolution kernel.
        stride: Stride of the 1D convolution kernel.
        dilation: Dilation of the 1D convolution kernel.

    Returns:
        A two-element tuple of left and right padding amounts such that the
        output length after applying a 1D convolution with parameters
        ``kernel_size``, ``stride``, and ``dilation`` to a tensor padded with
        these amounts is equal to ``math.ceil(float(length) / stride)``.

    Raises:
        ValueError: if ``length <= 0`` or ``kernel_size <= 0`` or
            ``stride <= 0`` or ``dilation <= 0``.

    Example:
        >>> length = 100
        >>> kernel_size = 5
        >>> conv = torch.nn.Conv1d(
        ...     in_channels=1,
        ...     out_channels=1,
        ...     kernel_size=kernel_size
        ... )
        >>> # create input tensor with batch=1, channels=1
        >>> x = torch.empty([1, 1, length]).normal_()
        >>> # add padding
        >>> padding = pad_same(length, kernel_size)
        >>> padding
        (2, 2)
        >>> x = torch.nn.functional.pad(x, padding)
        >>> # compute output, due to padding and stride = 1 the output length
        >>> # is equal to the input length
        >>> list(conv(x).size())
        [1, 1, 100]
    """
    if length <= 0:
        raise ValueError(f"length={length} must be > 0")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size={kernel_size} must be > 0")
    if stride <= 0:
        raise ValueError(f"stride={stride} must be > 0")
    if dilation <= 0:
        raise ValueError(f"dilation={dilation} must be > 0")
    effective_ks = dilation * (kernel_size - 1) + 1
    # this is equivalent to ``ceil(length / stride) - 1`` but only uses integer
    # operations
    out_dim = (length + stride - 1) // stride
    pad = int(stride * (out_dim) - 1) + effective_ks - length
    pad_l = pad // 2
    pad_r = pad - pad_l
    return pad_l, pad_r


def out_lens(
    seq_lens: torch.Tensor,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
) -> torch.Tensor:
    """Returns a tensor of sequence lengths after applying convolution.

    Args:
        seq_lens: A :py:class:`torch.Tensor` of size ``[batch]`` of
            sequence lengths.

        kernel_size: The size of the convolution kernel.

        stride: The stride of the convolution.

        dilation: The dilation of the convolution kernel.

        padding: The total amount of padding applied to each sequence.

    Returns:
        A :py:class:`torch.Tensor` of size ``[batch]`` of output sequence
        lengths after the padding and convolution are applied.
    """
    in_dtype = seq_lens.dtype
    seq_lens = seq_lens.float()
    seq_lens += padding
    seq_lens -= dilation * (kernel_size - 1) + 1
    seq_lens /= stride
    seq_lens += 1
    return seq_lens.floor().to(in_dtype)


class MaskConv1d(torch.nn.Conv1d):
    """Applies a 1D convolution over an input signal with a given length.

    This wrapper ensures the sequence length information is correctly used by
    the :py:class:`torch.nn.Conv1d`. Each signal in a batch has an associated
    length. For each signal the layer first sets all values at indices greater
    than the corresponding length to zero. It then concatenates the padding, if
    any, before applying the :py:class:`torch.nn.Conv1d`.

    Args:
        in_channels: See :py:class:`torch.nn.Conv1d`.

        out_channels: See :py:class:`torch.nn.Conv1d`.

        kernel_size: See :py:class:`torch.nn.Conv1d`.

        stride: See :py:class:`torch.nn.Conv1d`.

        padding_mode: The :py:class:`PaddingMode` to apply.

        dilation: See :py:class:`torch.nn.Conv1d`.

        groups: See :py:class:`torch.nn.Conv1d`.

        bias: See :py:class:`torch.nn.Conv1d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding_mode: PaddingMode = PaddingMode.NONE,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.padding_mode = padding_mode
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            super().cuda()

    def _pad(
        self, acts: torch.Tensor, seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads the input activations."""
        batch, channels, max_seq_len = acts.size()

        if self.padding_mode == PaddingMode.NONE:
            pad = (0, 0)
        elif self.padding_mode == PaddingMode.SAME:
            pad = pad_same(
                length=max_seq_len,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                dilation=self.dilation[0],
            )
            acts = torch.nn.functional.pad(acts, pad)
        else:
            raise ValueError(f"unknown padding mode {self.padding_mode}")

        seq_lens = out_lens(
            seq_lens,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            padding=sum(pad),
        )
        return acts, seq_lens

    def _mask_(self, acts: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """Sets all elements longer than each signals length to zero."""
        max_seq_len = acts.size(2)

        mask = (
            torch.arange(max_seq_len)
            .to(seq_lens.device)
            .expand(len(seq_lens), max_seq_len)
        )
        mask = mask >= seq_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).type(torch.bool).to(device=acts.device)

        acts.masked_fill_(mask, 0)
        del mask

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the module to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch, in_channels,
                in_seq_len]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying the module to ``x[0]``. It must have size ``[batch,
            out_channels, out_seq_len]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())

        acts, seq_lens = x

        self._mask_(acts, seq_lens)

        acts, seq_lens = self._pad(acts, seq_lens)

        acts = super().forward(acts)

        return acts, seq_lens

    def extra_repr(self) -> str:
        return super().extra_repr() + f", padding_mode={self.padding_mode}"


class MaskConv2d(torch.nn.Conv2d):
    """Applies a 2D convolution over an input signal with a given length.

    This wrapper ensures the sequence length information is correctly used by
    the :py:class:`torch.nn.Conv2d`. Each signal in a batch has an associated
    length. For each signal the layer first sets all values at indices greater
    than the corresponding length to zero. It then concatenates the padding, if
    any, before applying the :py:class:`torch.nn.Conv1d`.

    Args:
        in_channels: See :py:class:`torch.nn.Conv1d`.

        out_channels: See :py:class:`torch.nn.Conv1d`.

        kernel_size: See :py:class:`torch.nn.Conv1d`.

        stride: See :py:class:`torch.nn.Conv1d`.

        padding_mode: The :py:class:`PaddingMode` to apply.

        dilation: See :py:class:`torch.nn.Conv1d`.

        groups: See :py:class:`torch.nn.Conv1d`.

        bias: See :py:class:`torch.nn.Conv1d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding_mode: PaddingMode = PaddingMode.NONE,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.padding_mode = padding_mode
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            super().cuda()

    def _pad(
        self, acts: torch.Tensor, seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads the input activations."""
        batch, channels, features, max_seq_len = acts.size()

        if self.padding_mode == PaddingMode.NONE:
            pad_len = (0, 0)
        elif self.padding_mode == PaddingMode.SAME:
            pad_len = pad_same(
                length=max_seq_len,
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                dilation=self.dilation[1],
            )
            pad_features = pad_same(
                length=features,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                dilation=self.dilation[0],
            )
            acts = torch.nn.functional.pad(acts, pad_len + pad_features)
        else:
            raise ValueError(f"unknown padding mode {self.padding_mode}")

        seq_lens = out_lens(
            seq_lens,
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            dilation=self.dilation[1],
            padding=sum(pad_len),
        )
        return acts, seq_lens

    def _mask_(self, acts: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """Sets all elements longer than each signals length to zero."""
        max_seq_len = acts.size(3)

        mask = (
            torch.arange(max_seq_len)
            .to(seq_lens.device)
            .expand(len(seq_lens), max_seq_len)
        )
        mask = mask >= seq_lens.unsqueeze(1)
        mask = (
            mask.unsqueeze(1)  # add channels and features dims, these will be
            .unsqueeze(1)  # broadcast so OK to be set to 1
            .type(torch.bool)
            .to(device=acts.device)
        )

        acts.masked_fill_(mask, 0)
        del mask

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the module to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch, in_channels,
                in_features, in_seq_len]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying the module to ``x[0]``. It must have size ``[batch,
            out_channels, out_features, out_seq_len]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())

        acts, seq_lens = x

        self._mask_(acts, seq_lens)

        acts, seq_lens = self._pad(acts, seq_lens)

        acts = super().forward(acts)

        return acts, seq_lens

    def extra_repr(self) -> str:
        return super().extra_repr() + f", padding_mode={self.padding_mode}"
