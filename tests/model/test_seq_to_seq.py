import copy
import warnings
from tempfile import NamedTemporaryFile

import pytest
import torch
from hypothesis import given
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.load import load_seq_to_seq

from tests.builders.test_task_config import build_and_check_task_config
from tests.protos.test_task_config import task_configs
from tests.utils.utils import check_state_dicts_match

# Tests -----------------------------------------------------------------------


@given(seq_to_seq_cfg=task_configs())
def test_seq_to_seq_state_dict_correctly_saved_and_loaded(
    seq_to_seq_cfg: task_config_pb2.TaskConfig,
) -> None:
    """Test that SeqToSeq task state_dict is correctly saved and loaded.

    This is also an end to end test for the task_config builder.
    """
    if seq_to_seq_cfg is None:
        warnings.warn("Skipping test as invalid task_config proto drawn.")
        return
    seq_to_seq = build_and_check_task_config(seq_to_seq_cfg)
    if seq_to_seq is None:
        return
    orig_state_dict = copy.deepcopy(seq_to_seq.state_dict())

    # save state_dict
    with NamedTemporaryFile() as f:
        fp = f.name
        torch.save(orig_state_dict, fp)

        # Alter state_dict so that it != orig_state_dict
        for param_group in seq_to_seq.optim.param_groups:
            param_group["lr"] += 0.2
        seq_to_seq.lr_scheduler.step_freq += 1
        if seq_to_seq.lr_scheduler.num_warmup_steps is not None:
            seq_to_seq.lr_scheduler.num_warmup_steps += 1
        with torch.no_grad():
            for name, param in seq_to_seq.model.named_parameters():
                param += 1

        # Ensure changes were successful
        with pytest.raises(AssertionError):
            check_state_dicts_match(seq_to_seq.state_dict(), orig_state_dict)
        with pytest.raises(AssertionError):
            check_state_dicts_match(
                seq_to_seq.model.state_dict(), orig_state_dict["model"]
            )
        with pytest.raises(AssertionError):
            check_state_dicts_match(
                seq_to_seq.optim.state_dict(), orig_state_dict["optim"]
            )
        with pytest.raises(AssertionError):
            check_state_dicts_match(
                seq_to_seq.lr_scheduler.state_dict(),
                orig_state_dict["lr_scheduler"],
            )

        # load orig_state_dict back into seq_to_seq
        training_state = load_seq_to_seq(seq_to_seq, fp)

    model_sd = seq_to_seq.state_dict()["model"]
    expected_model_sd = orig_state_dict["model"]
    optim_sd = seq_to_seq.state_dict()["optim"]
    expected_optim_sd = orig_state_dict["optim"]
    lr_scheduler_sd = seq_to_seq.state_dict()["lr_scheduler"]
    expected_lr_scheduler_sd = orig_state_dict["lr_scheduler"]

    check_state_dicts_match(model_sd, expected_model_sd)
    check_state_dicts_match(optim_sd, expected_optim_sd)
    check_state_dicts_match(lr_scheduler_sd, expected_lr_scheduler_sd)
    assert training_state == {}
