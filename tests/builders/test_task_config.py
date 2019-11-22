import warnings

import torch
from hypothesis import given
from hypothesis import settings
from myrtlespeech.builders.task_config import build
from myrtlespeech.protos import task_config_pb2

from tests.protos.test_task_config import task_configs


# Tests -----------------------------------------------------------------------


@given(task_cfg=task_configs())
@settings(deadline=3000)
def test_build_returns(task_cfg: task_config_pb2.TaskConfig) -> None:
    """Test that build returns when called."""
    try:
        model, epochs, train_loader, eval_loader = build(task_cfg)
        if model is not None:
            assert isinstance(model, torch.nn.Module)
        else:
            warnings.warn(
                "Not checking if model is returned. Remove above `if` \
            statement once tests/protos/test_speech_to_text.py has had \
            exception handling removed"
            )

        assert isinstance(epochs, int)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(eval_loader, torch.utils.data.DataLoader)
        warnings.warn("TaskConfig only built and not checked if correct")
    except ValueError as e:
        if str(e) == "unsupported model None":
            warnings.warn(f"Caught error {e}.")
        else:
            raise e
