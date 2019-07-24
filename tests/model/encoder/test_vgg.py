from typing import List, Optional, Tuple

import hypothesis.strategies as st
import torch
import pytest
from hypothesis import given, assume

from myrtlespeech.model.seq_len_wrapper import SeqLenWrapper
from myrtlespeech.model.encoder.vgg import (
    VGGConfig,
    make_layers,
    vgg_output_size,
)


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def random_cfg(draw) -> st.SearchStrategy[VGGConfig]:
    """Returns a SearchStrategy for a valid VGGConfig."""
    cfg = draw(
        st.lists(
            st.one_of(
                st.integers(min_value=1, max_value=128), st.sampled_from(["M"])
            ),
            min_size=0,
            max_size=30,
        )
    )
    return cfg


@st.composite
def random_invalid_cfg(draw) -> st.SearchStrategy[List]:
    """Returns a SearchStrategy for an invalid VGGConfig."""
    strategies = [
        st.integers(min_value=1, max_value=128),
        st.floats(),
        st.characters(),
    ]
    invalid_cfg = draw(st.lists(st.one_of(strategies), min_size=0, max_size=30))
    assume(
        not all(
            [
                isinstance(cfg_val, int) or cfg_val == "M"
                for cfg_val in invalid_cfg
            ]
        )
    )

    return invalid_cfg


@st.composite
def vgg_input_sizes(
    draw
) -> st.SearchStrategy[Tuple[torch.nn.Sequential, torch.Size]]:
    """Returns a SearchStrategy for modules from VGGConfig's and input_sizes."""
    input_size = draw(st.lists(st.integers(1, 32), min_size=4, max_size=4))
    input_size = torch.Size(input_size)

    cfg = draw(random_cfg())
    batch_norm = draw(st.booleans())
    use_output_from_block = draw(st.integers(1, 5))
    vgg = make_layers(
        cfg=cfg,
        in_channels=input_size[1],
        batch_norm=batch_norm,
        use_output_from_block=use_output_from_block,
    )

    return vgg, input_size


# Tests -----------------------------------------------------------------------

# make_layers ---------------------------------------


@given(
    cfg=random_cfg(),
    in_channels=st.integers(min_value=1, max_value=128),
    batch_norm=st.booleans(),
    initialize_weights=st.booleans(),
    use_output_from_block=st.one_of(st.none(), st.integers(1, 5)),
    seq_len_support=st.booleans(),
)
def test_make_layers_returns_correct_module_structure_for_valid_cfg(
    cfg: VGGConfig,
    in_channels: int,
    batch_norm: bool,
    initialize_weights: bool,
    use_output_from_block: Optional[int],
    seq_len_support: bool,
) -> None:
    """Ensures Module returned by ``make_layers`` has correct structure."""
    module = make_layers(
        cfg,
        in_channels,
        batch_norm,
        initialize_weights,
        use_output_from_block,
        seq_len_support,
    )

    if seq_len_support:
        assert isinstance(module, SeqLenWrapper)
        module = module.module
    else:
        assert isinstance(module, torch.nn.Sequential)

    m_idx = 0  # iterate through torch.nn.Sequential module using index from 0
    n_blocks = 0  # track blocks to test use_output_from_block
    for cfg_val in cfg:
        if use_output_from_block and n_blocks >= use_output_from_block:
            # if layers exist after seeing use_output_from_block max pools then
            # layers not truncated
            with pytest.raises(IndexError):
                module[m_idx]
            return

        m = module[m_idx]
        if cfg_val == "M":
            assert isinstance(m, torch.nn.MaxPool2d)
            assert m.kernel_size == 2
            assert m.stride == 2
            n_blocks += 1
        else:
            assert isinstance(m, torch.nn.Conv2d)
            assert m.in_channels == in_channels
            in_channels = m.out_channels
            assert m.out_channels == cfg_val
            assert m.kernel_size == (3, 3)
            assert m.padding == (1, 1)

            if batch_norm:
                m_idx += 1
                assert isinstance(module[m_idx], torch.nn.BatchNorm2d)

            m_idx += 1
            m = module[m_idx]
            assert isinstance(m, torch.nn.ReLU)
            assert m.inplace is True

        m_idx += 1


@given(
    cfg=random_cfg(),
    in_channels=st.integers(min_value=1, max_value=128),
    batch_norm=st.booleans(),
    initialize_weights=st.booleans(),
    use_output_from_block=st.one_of(st.none(), st.integers(1, 5)),
    batch_size=st.integers(min_value=1, max_value=18),
    features=st.integers(min_value=1, max_value=32),
    max_seq_len=st.integers(min_value=1, max_value=99),
)
def test_make_layers_returned_module_returns_correct_seq_len(
    cfg: VGGConfig,
    in_channels: int,
    batch_norm: bool,
    initialize_weights: bool,
    use_output_from_block: Optional[int],
    batch_size: int,
    features: int,
    max_seq_len: int,
) -> None:
    """Ensures Module returns correct seq_lens when seq_len_support=True."""
    module = make_layers(
        cfg,
        in_channels,
        batch_norm,
        initialize_weights,
        use_output_from_block,
        seq_len_support=True,
    )

    # create input and seq_lens tensors
    input_size = [batch_size, in_channels, features, max_seq_len]
    tensor = torch.empty(input_size, requires_grad=False).normal_()

    in_seq_lens = torch.randint(
        low=1,
        high=max_seq_len + 1,
        size=[batch_size],
        dtype=torch.int32,
        requires_grad=False,
    )

    # compute output and check
    _, act_seq_lens = module(tensor, seq_lens=in_seq_lens)

    exp_seq_lens = []
    for seq_len in in_seq_lens:
        exp_seq_lens.append(
            vgg_output_size(module, torch.Size([1, 1, 1, seq_len]))[3]
        )

    assert act_seq_lens.tolist() == exp_seq_lens


@given(cfg=random_invalid_cfg())
def test_make_layers_raises_value_error_for_invalid_cfg(cfg: List) -> None:
    """Ensures ValueError raised when cfg is invalid."""
    with pytest.raises(ValueError):
        make_layers(cfg)


@given(
    cfg=random_cfg(),
    use_output_from_block=st.one_of(
        st.integers(-1000, 0), st.integers(6, 1000)
    ),
)
def test_make_layers_raises_value_error_for_invalid_use_output_from_block(
    cfg: VGGConfig, use_output_from_block: int
) -> None:
    """Ensures ValueError raised when use_output_from_block invalid."""
    with pytest.raises(ValueError):
        make_layers(cfg, use_output_from_block=use_output_from_block)


# vgg_ouput_size ------------------------------------


@given(vgg_input_size=vgg_input_sizes())
def test_vgg_output_size_returns_correct_size(
    vgg_input_size: Tuple[torch.nn.Sequential, torch.Size]
) -> None:
    """Ensures vgg_output_size matches actual module output size."""
    vgg, input_size = vgg_input_size
    # BN expects more than 1 value per channel when training or err thrown
    # https://github.com/pytorch/pytorch/blob/34aee933f9f38f80adaa38b52f4cd5a59cb47e48/torch/nn/functional.py#L1704
    for idx, module in enumerate(vgg.modules()):
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_input_size = vgg_output_size(vgg[: max(0, idx - 1)], input_size)
            assume(bn_input_size[2] > 1 and bn_input_size[3] > 1)
    x = torch.empty(input_size).normal_()
    assert vgg_output_size(vgg, input_size) == vgg(x).size()
