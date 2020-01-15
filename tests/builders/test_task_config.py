import warnings
from typing import Optional

import torch
from myrtlespeech.builders.task_config import build
from myrtlespeech.protos import task_config_pb2

# Utilities -------------------------------------------------------------------


def build_and_check_task_config(
    task_cfg: task_config_pb2.TaskConfig,
) -> Optional[torch.nn.Module]:
    """Helper to check TaskConfig is built correctly."""
    seq_to_seq = None
    if task_cfg.HasField("speech_to_text"):
        seq_to_seq, epochs, train_loader, eval_loader = build(task_cfg)
        assert isinstance(seq_to_seq, torch.nn.Module)
        assert isinstance(epochs, int)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(eval_loader, torch.utils.data.DataLoader)
        warnings.warn("TaskConfig only built and not checked if correct")
    else:
        warnings.warn(
            "Invalid proto drawn w/o a speech_to_text config."
            "TOD0: Remove this exception handling once the hack in "
            "tests/protos/test_speech_to_text.py has been removed."
        )
    return seq_to_seq
