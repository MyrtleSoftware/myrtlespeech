from typing import List, Optional

import hypothesis.strategies as st
import torch
import pytest
from hypothesis import given, assume

from myrtlespeech.model.encoder.vgg import VGGConfig, make_layers


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


# Tests -----------------------------------------------------------------------


@given(
    cfg=random_cfg(),
    in_channels=st.integers(min_value=1, max_value=128),
    batch_norm=st.booleans(),
    initialize_weights=st.booleans(),
    use_output_from_block=st.one_of(st.none(), st.integers(1, 5)),
)
def test_make_layers_returns_correct_module_structure_for_valid_cfg(
    cfg: VGGConfig,
    in_channels: int,
    batch_norm: bool,
    initialize_weights: bool,
    use_output_from_block: Optional[int],
) -> None:
    """Ensures Module returned by ``make_layers`` has correct structure."""
    module = make_layers(
        cfg, in_channels, batch_norm, initialize_weights, use_output_from_block
    )

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
