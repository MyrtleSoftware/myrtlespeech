from typing import Tuple

import pytest
from hypothesis import given
from myrtlespeech.builders.pre_process_step import build
from myrtlespeech.data.preprocess import MFCC
from myrtlespeech.protos import pre_process_step_pb2
from myrtlespeech.run.stage import Stage

from tests.protos.test_pre_process_step import pre_process_steps


# Utilities -------------------------------------------------------------------


def pre_process_step_match_cfg(
    step: Tuple[MFCC, Stage], step_cfg: pre_process_step_pb2.PreProcessStep
) -> None:
    """Ensures preprocessing step matches protobuf configuration."""
    assert step[1] == Stage(step_cfg.stage)

    step_str = step_cfg.WhichOneof("pre_process_step")

    if step_str == "mfcc":
        assert step[0].numcep == step_cfg.mfcc.numcep
        assert step[0].winlen == step_cfg.mfcc.winlen
        assert step[0].winstep == step_cfg.mfcc.winstep
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
