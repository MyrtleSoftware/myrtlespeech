from typing import Tuple
from typing import Union

import pytest
from hypothesis import given
from myrtlespeech.builders.pre_process_step import build
from myrtlespeech.data.preprocess import AddContextFrames
from myrtlespeech.data.preprocess import LogMelFB
from myrtlespeech.data.preprocess import Standardize
from myrtlespeech.protos import pre_process_step_pb2
from myrtlespeech.run.stage import Stage
from torchaudio.transforms import MFCC

from tests.protos.test_pre_process_step import pre_process_steps


# Utilities -------------------------------------------------------------------


def pre_process_step_match_cfg(
    step: Tuple[Union[MFCC, Standardize], Stage],
    step_cfg: pre_process_step_pb2.PreProcessStep,
) -> None:
    """Ensures preprocessing step matches protobuf configuration."""
    assert step[1] == Stage(step_cfg.stage)

    step_str = step_cfg.WhichOneof("pre_process_step")

    if step_str == "mfcc":
        assert isinstance(step[0], MFCC)
        assert step[0].n_mfcc == step_cfg.mfcc.n_mfcc
        assert step[0].MelSpectrogram.win_length == step_cfg.mfcc.win_length
        assert step[0].MelSpectrogram.hop_length == step_cfg.mfcc.hop_length
    elif step_str == "lmfb":
        assert isinstance(step[0], LogMelFB)
        assert step[0].MelSpectrogram.n_mels == step_cfg.lmfb.n_mels
        assert step[0].MelSpectrogram.win_length == step_cfg.lmfb.win_length
        assert step[0].MelSpectrogram.hop_length == step_cfg.lmfb.hop_length
    elif step_str == "standardize":
        assert isinstance(step[0], Standardize)
    elif step_str == "context_frames":
        assert isinstance(step[0], AddContextFrames)
        assert step[0].n_context == step_cfg.context_frames.n_context
    else:
        raise ValueError(f"unknown pre_process_step {step_str}")


# Tests -----------------------------------------------------------------------


@given(step_cfg=pre_process_steps())
def test_build_returns_correct_pre_process_step_with_valid_params(
    step_cfg: pre_process_step_pb2.PreProcessStep
) -> None:
    """Test that build returns the correct preprocess step with valid params."""
    step = build(step_cfg)
    pre_process_step_match_cfg(step, step_cfg)


@given(step_cfg=pre_process_steps())
def test_unknown_pre_process_step_raises_value_error(
    step_cfg: pre_process_step_pb2.PreProcessStep
) -> None:
    """Ensures ValueError is raised when pre_process_step not supported.

    This can occur when the protobuf is updated and build is not.
    """
    step_cfg.ClearField(step_cfg.WhichOneof("pre_process_step"))
    with pytest.raises(ValueError):
        build(step_cfg)
