import warnings
from typing import Tuple

import torch
from hypothesis import given
from myrtlespeech.builders.task_config import build
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import task_config_pb2

from tests.protos.test_task_config import task_configs

# Utilities -------------------------------------------------------------------


def build_and_check_task_config(
    task_cfg: task_config_pb2.TaskConfig,
) -> Tuple[
    SeqToSeq, int, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    """Helper to check TaskConfig is built correctly."""
    seq_to_seq, epochs, train_loader, eval_loader = build(task_cfg)
    assert isinstance(seq_to_seq, torch.nn.Module)
    assert isinstance(epochs, int)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(eval_loader, torch.utils.data.DataLoader)
    warnings.warn("TaskConfig only built and not checked if correct")
    return seq_to_seq, epochs, train_loader, eval_loader


# Tests -----------------------------------------------------------------------


@given(task_cfg=task_configs())
def test_build_returns(task_cfg: task_config_pb2.TaskConfig) -> None:
    """Test that build returns when called."""
    build_and_check_task_config(task_cfg)
