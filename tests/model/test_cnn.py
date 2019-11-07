import math
from typing import Tuple

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from myrtlespeech.model.cnn import MaskConv1d
from myrtlespeech.model.cnn import MaskConv2d
from myrtlespeech.model.cnn import out_lens
from myrtlespeech.model.cnn import pad_same
from myrtlespeech.model.cnn import PaddingMode


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def padding_modes(draw) -> st.SearchStrategy[PaddingMode]:
    """A strategy for :py:class:`PaddingMode`."""
    return draw(st.sampled_from(PaddingMode))


st.register_type_strategy(PaddingMode, padding_modes)


@st.composite
def mask_conv1ds(draw) -> st.SearchStrategy[MaskConv1d]:
    """Returns a SearchStrategy for MaskConv1d."""
    in_channels = draw(st.integers(1, 8))
    out_channels = draw(st.integers(1, 8))
    kernel_size = draw(st.integers(1, 5))
    stride = draw(st.integers(1, 5))
    dilation = draw(st.integers(1, 5))
    padding_mode = draw(padding_modes())

    return MaskConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding_mode=padding_mode,
        dilation=dilation,
    )


@st.composite
def mask_conv1d_valid_inputs(
    draw,
) -> st.SearchStrategy[Tuple[MaskConv1d, Tuple[torch.Tensor, torch.Tensor]]]:
    """Returns a SearchStrategy for MaskConv1d and valid input."""
    mask_conv1d = draw(mask_conv1ds())
    batch_size = draw(st.integers(1, 8))

    # PyTorch conv1d requires input size to be greater than effective kernel
    # size. Force this to be True.
    if mask_conv1d._padding_mode == PaddingMode.NONE:
        # NONE requires setting the minimum to the effective kernel size.
        effective_ks = (
            mask_conv1d.dilation[0] * (mask_conv1d.kernel_size[0] - 1) + 1
        )
        seq_len = draw(st.integers(effective_ks, effective_ks + 256))
    elif mask_conv1d._padding_mode == PaddingMode.SAME:
        # SAME padding should sort this for us.
        seq_len = draw(st.integers(1, 257))
    else:
        raise ValueError(f"unknown PaddingMode {mask_conv1d._padding_mode}")

    tensor = torch.empty(
        [batch_size, mask_conv1d.in_channels, seq_len]
    ).normal_()

    seq_lens = torch.tensor(
        draw(
            st.lists(
                st.integers(1, seq_len + 1),
                min_size=batch_size,
                max_size=batch_size,
            )
        )
    )
    return mask_conv1d, (tensor, seq_lens)


@st.composite
def mask_conv2ds(draw) -> st.SearchStrategy[MaskConv2d]:
    """Returns a SearchStrategy for MaskConv2d."""
    in_channels = draw(st.integers(1, 8))
    out_channels = draw(st.integers(1, 8))
    kernel_size = draw(st.integers(1, 5))
    stride = draw(st.integers(1, 5))
    dilation = draw(st.integers(1, 5))
    padding_mode = draw(padding_modes())

    return MaskConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding_mode=padding_mode,
        dilation=dilation,
    )


@st.composite
def mask_conv2d_valid_inputs(
    draw,
) -> st.SearchStrategy[Tuple[MaskConv2d, Tuple[torch.Tensor, torch.Tensor]]]:
    """Returns a SearchStrategy for MaskConv2d and valid input."""
    mask_conv2d = draw(mask_conv2ds())
    batch_size = draw(st.integers(1, 8))

    # PyTorch conv1d requires input size to be greater than effective kernel
    # size. Force this to be True.
    if mask_conv2d._padding_mode == PaddingMode.NONE:
        effective_ks = (
            mask_conv2d.dilation[0] * (mask_conv2d.kernel_size[0] - 1) + 1
        )
        features = draw(st.integers(effective_ks, effective_ks + 32))
        # NONE requires setting the minimum to the effective kernel size.
        effective_ks = (
            mask_conv2d.dilation[1] * (mask_conv2d.kernel_size[1] - 1) + 1
        )
        seq_len = draw(st.integers(effective_ks, effective_ks + 32))
    elif mask_conv2d._padding_mode == PaddingMode.SAME:
        # SAME padding should sort this for us.
        features = draw(st.integers(1, 257))
        seq_len = draw(st.integers(1, 257))
    else:
        raise ValueError(f"unknown PaddingMode {mask_conv2d._padding_mode}")

    tensor = torch.empty(
        [batch_size, mask_conv2d.in_channels, features, seq_len]
    ).normal_()

    seq_lens = torch.tensor(
        draw(
            st.lists(
                st.integers(1, seq_len + 1),
                min_size=batch_size,
                max_size=batch_size,
            )
        )
    )
    return mask_conv2d, (tensor, seq_lens)


@st.composite
def conv1ds(draw) -> st.SearchStrategy[torch.nn.Conv1d]:
    """Returns a SearchStrategy for Conv1d."""
    in_channels = draw(st.integers(1, 8))
    out_channels = draw(st.integers(1, 8))
    kernel_size = draw(st.integers(1, 41))
    stride = draw(st.integers(1, 5))
    dilation = draw(st.integers(1, 5))

    return torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=dilation,
    )


@st.composite
def conv1d_valid_inputs(
    draw,
) -> st.SearchStrategy[Tuple[torch.nn.Conv1d, torch.Tensor]]:
    """Returns a SearchStrategy for Conv1d and valid input."""
    conv1d = draw(conv1ds())
    batch_size = draw(st.integers(1, 8))
    seq_len = draw(st.integers(1, 256))
    tensor = torch.empty([batch_size, conv1d.in_channels, seq_len]).normal_()
    return conv1d, tensor


# Tests -----------------------------------------------------------------------

# pad_same ------------------------------


@given(conv1d_valid_input=conv1d_valid_inputs())
def test_pad_same_output_has_correct_size_after_conv(
    conv1d_valid_input: Tuple[torch.nn.Conv1d, torch.Tensor]
) -> None:
    """Ensures output size after applying [pad_same, Conv1d](x) is correct."""
    conv1d, tensor = conv1d_valid_input
    batch_size, in_channels, seq_len = tensor.size()

    pad = pad_same(
        length=seq_len,
        kernel_size=conv1d.kernel_size[0],
        stride=conv1d.stride[0],
        dilation=conv1d.dilation[0],
    )

    tensor = torch.nn.functional.pad(tensor, pad)

    out = conv1d(tensor)

    assert out.size(2) == math.ceil(float(seq_len) / conv1d.stride[0])


@given(
    length=st.integers(-100, 100),
    kernel_size=st.integers(-100, 100),
    stride=st.integers(-100, 100),
    dilation=st.integers(-100, 100),
)
def test_pad_same_raises_value_error_invalid_parameters(
    length: int, kernel_size: int, stride: int, dilation: int
) -> None:
    """Ensures pad_same raises ValueError when called with invalid params."""
    assume(length <= 0 or kernel_size <= 0 or stride <= 0 or dilation <= 0)
    with pytest.raises(ValueError):
        pad_same(length, kernel_size, stride, dilation)


# out_lens ------------------------------


@given(data=st.data(), conv1d=conv1ds())
def test_mask_conv_1d_out_lens(data, conv1d: torch.nn.Conv1d) -> None:
    """Ensures out_lens returns correct output sequence lengths."""
    # PyTorch conv1d requires input size to be greater than effective kernel
    # size. Force this to be True.
    effective_ks = conv1d.dilation[0] * (conv1d.kernel_size[0] - 1) + 1

    padding = data.draw(st.integers(0, effective_ks + 100))

    seq_lens = data.draw(
        st.lists(
            elements=st.integers(max(0, effective_ks - padding), 256),
            min_size=0,
            max_size=32,
        )
    )

    exp = []
    for seq_len in seq_lens:
        out = conv1d(torch.empty([1, conv1d.in_channels, seq_len + padding]))
        exp.append(out.size(2))
    act = list(
        out_lens(
            seq_lens=torch.tensor(seq_lens),
            kernel_size=conv1d.kernel_size[0],
            stride=conv1d.stride[0],
            dilation=conv1d.dilation[0],
            padding=padding,
        )
    )

    assert exp == act


# MaskConv1d ----------------------------


@given(mask_conv1d_input=mask_conv1d_valid_inputs())
@settings(deadline=3000)
def test_mask_conv_output_size_and_seq_lens(
    mask_conv1d_input: Tuple[MaskConv1d, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """Ensures the MaskConv1d output size and seq_len values are correct."""
    mask_conv1d, inputs = mask_conv1d_input
    batch, channels, seq_len = inputs[0].size()

    out_tensor, out_seq_lens = mask_conv1d(inputs)
    out_batch, out_channels, out_seq_len = out_tensor.size()
    assert out_batch == len(out_seq_lens) == batch
    assert out_channels == mask_conv1d.out_channels

    # test out_seq_len dimension of tensor output and set padding value ready
    # for testing out_seq_lens
    if mask_conv1d._padding_mode == PaddingMode.NONE:
        padding = (0, 0)
        # out_lens function returns correct expected length if the tests pass
        exp_len = out_lens(
            seq_lens=torch.tensor(seq_len),
            kernel_size=mask_conv1d.kernel_size[0],
            stride=mask_conv1d.stride[0],
            dilation=mask_conv1d.dilation[0],
            padding=0,
        ).item()
        assert out_seq_len == exp_len
    elif mask_conv1d._padding_mode == PaddingMode.SAME:
        padding = pad_same(
            length=seq_len,
            kernel_size=mask_conv1d.kernel_size[0],
            stride=mask_conv1d.stride[0],
            dilation=mask_conv1d.dilation[0],
        )
        # by definition of SAME padding
        assert out_seq_len == math.ceil(float(seq_len) / mask_conv1d.stride[0])
    else:
        raise ValueError(f"unknown PaddingMode {mask_conv1d._padding_mode}")

    # test out_seq_lens
    exp_out_seq_lens = out_lens(
        seq_lens=inputs[1],
        kernel_size=mask_conv1d.kernel_size[0],
        stride=mask_conv1d.stride[0],
        dilation=mask_conv1d.dilation[0],
        padding=sum(padding),
    ).to(out_seq_lens.device)
    assert torch.all(out_seq_lens == exp_out_seq_lens)


@given(mask_conv1d_input=mask_conv1d_valid_inputs())
def test_all_gradients_computed_for_all_model_parameters(
    mask_conv1d_input: Tuple[MaskConv1d, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    mask_conv1d, inputs = mask_conv1d_input

    # forward pass
    out = mask_conv1d(inputs)

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    # check all parameters have gradient
    for name, p in mask_conv1d.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


# MaskConv2d ----------------------------


@given(mask_conv2d_input=mask_conv2d_valid_inputs())
def test_mask2d_conv_output_size_and_seq_lens(
    mask_conv2d_input: Tuple[MaskConv2d, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """Ensures the MaskConv2d output size and seq_len values are correct."""
    mask_conv2d, inputs = mask_conv2d_input
    batch, channels, features, seq_len = inputs[0].size()

    out_tensor, out_seq_lens = mask_conv2d(inputs)
    out_batch, out_channels, out_features, out_seq_len = out_tensor.size()
    assert out_batch == len(out_seq_lens) == batch
    assert out_channels == mask_conv2d.out_channels

    # test out_seq_len dimension of tensor output and set padding value ready
    # for testing out_seq_lens
    if mask_conv2d._padding_mode == PaddingMode.NONE:
        padding = (0, 0)
        # out_lens function returns correct expected length if the tests pass
        exp_features = out_lens(
            seq_lens=torch.tensor(features),
            kernel_size=mask_conv2d.kernel_size[0],
            stride=mask_conv2d.stride[0],
            dilation=mask_conv2d.dilation[0],
            padding=0,
        ).item()
        assert out_features == exp_features
        exp_len = out_lens(
            seq_lens=torch.tensor(seq_len),
            kernel_size=mask_conv2d.kernel_size[1],
            stride=mask_conv2d.stride[1],
            dilation=mask_conv2d.dilation[1],
            padding=0,
        ).item()
        assert out_seq_len == exp_len
    elif mask_conv2d._padding_mode == PaddingMode.SAME:
        padding = pad_same(
            length=features,
            kernel_size=mask_conv2d.kernel_size[0],
            stride=mask_conv2d.stride[0],
            dilation=mask_conv2d.dilation[0],
        )
        assert out_features == math.ceil(
            float(features) / mask_conv2d.stride[0]
        )

        padding = pad_same(
            length=seq_len,
            kernel_size=mask_conv2d.kernel_size[1],
            stride=mask_conv2d.stride[1],
            dilation=mask_conv2d.dilation[1],
        )
        # by definition of SAME padding
        assert out_seq_len == math.ceil(float(seq_len) / mask_conv2d.stride[1])
    else:
        raise ValueError(f"unknown PaddingMode {mask_conv2d._padding_mode}")

    # test out_seq_lens
    exp_out_seq_lens = out_lens(
        seq_lens=inputs[1],
        kernel_size=mask_conv2d.kernel_size[1],
        stride=mask_conv2d.stride[1],
        dilation=mask_conv2d.dilation[1],
        padding=sum(padding),
    ).to(out_seq_lens.device)
    assert torch.all(out_seq_lens == exp_out_seq_lens)


@given(mask_conv2d_input=mask_conv2d_valid_inputs())
def test_mask_conv2d_all_gradients_computed_for_all_model_parameters(
    mask_conv2d_input: Tuple[MaskConv2d, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    mask_conv2d, inputs = mask_conv2d_input

    # forward pass
    out = mask_conv2d(inputs)

    # backward pass using mean as proxy for an actual loss function
    loss = out[0].mean()
    loss.backward()

    # check all parameters have gradient
    for name, p in mask_conv2d.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
