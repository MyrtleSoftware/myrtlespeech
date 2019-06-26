import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.model.encoder.vgg import cfgs, make_layers
from myrtlespeech.protos import vgg_pb2
from myrtlespeech.builders.vgg import build_vgg
from tests.protos.test_vgg import vggs


# Tests -----------------------------------------------------------------------


@given(vgg_cfg=vggs(), input_channels=st.integers(min_value=1, max_value=8))
def test_build_vgg_returns_correct_module_structure(
    vgg_cfg: vgg_pb2.VGG, input_channels: int
) -> None:
    """Ensures Module returned by ``build_vgg`` has correct structure."""
    actual = build_vgg(vgg_cfg, input_channels)

    assert isinstance(actual, torch.nn.Sequential)

    c_map = {index: letter for letter, index in vgg_pb2.VGG.VGG_CONFIG.items()}
    expected = make_layers(
        cfg=cfgs[c_map[vgg_cfg.vgg_config]],
        in_channels=input_channels,
        batch_norm=vgg_cfg.batch_norm,
        initialize_weights=False,  # not checking weights
        use_output_from_block=vgg_cfg.use_output_from_block + 1,
    )

    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        assert type(act) == type(exp)
        if isinstance(act, torch.nn.Conv2d):
            assert act.in_channels == exp.in_channels
            assert act.out_channels == exp.out_channels
            assert act.kernel_size == exp.kernel_size
            assert act.stride == exp.stride
            assert act.padding == exp.padding
            assert act.dilation == exp.dilation
            assert act.groups == exp.groups
            assert act.padding_mode == exp.padding_mode
        elif isinstance(act, torch.nn.BatchNorm2d):
            assert act.num_features == exp.num_features
            assert act.eps == exp.eps
            assert act.momentum == exp.momentum
            assert act.affine == exp.affine
            assert act.track_running_stats == exp.track_running_stats
        elif isinstance(act, torch.nn.ReLU):
            assert act.inplace == exp.inplace
        elif isinstance(act, torch.nn.MaxPool2d):
            assert act.kernel_size == exp.kernel_size
            assert act.stride == exp.stride
            assert act.padding == exp.padding
            assert act.dilation == exp.dilation
            assert act.return_indices == exp.return_indices
            assert act.ceil_mode == exp.ceil_mode


@given(
    vgg_cfg=vggs(),
    input_channels=st.integers(1, 128),
    invalid_vgg_config=st.integers(0, 128),
)
def test_unknown_vgg_config_raises_value_error(
    vgg_cfg: vgg_pb2.VGG, input_channels: int, invalid_vgg_config: int
) -> None:
    """Ensures ValueError is raised when vgg_config not supported.

    This can occur when the protobuf is updated and build_vgg is not.
    """
    assume(invalid_vgg_config not in vgg_pb2.VGG.VGG_CONFIG.values())
    vgg_cfg.vgg_config = invalid_vgg_config  # type: ignore
    with pytest.raises(ValueError):
        build_vgg(vgg_cfg, input_channels=input_channels)
