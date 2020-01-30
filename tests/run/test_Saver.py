import copy
from tempfile import TemporaryDirectory

import pytest
import torch
from hypothesis import given
from hypothesis import settings
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.callbacks.callback import CallbackHandler
from myrtlespeech.run.run import Saver
from myrtlespeech.run.train import fit

from tests.builders.test_task_config import build_and_check_task_config
from tests.protos.test_task_config import task_configs
from tests.utils.utils import check_state_dicts_match

# Utilities -------------------------------------------------------------------


def change_seq_to_seq_state_dict(seq_to_seq: SeqToSeq):
    """Alters SeqToSeq state_dict.

    This should ensure that calling this function multiple times should result
    in a unique state_dict for **every** successive call.
    """
    with torch.no_grad():
        for name, param in seq_to_seq.model.named_parameters():
            param += 1.0
    if seq_to_seq.optim:
        for param_group in seq_to_seq.optim.param_groups:
            param_group["lr"] += 0.2
    if seq_to_seq.lr_scheduler:
        seq_to_seq.lr_scheduler.base_lrs = [
            x + 0.1 for x in seq_to_seq.lr_scheduler.base_lrs
        ]
        seq_to_seq.lr_scheduler._step_count += 2


# Tests -----------------------------------------------------------------------


@settings(deadline=3000)
@given(seq_to_seq_cfg=task_configs())
def test_seq_to_seq_correctly_built_saved_and_loaded(
    seq_to_seq_cfg: task_config_pb2.TaskConfig,
) -> None:
    """Test that SeqToSeq is correctly built + saved + loaded."""
    seq_to_seq, epochs, train_loader, _ = build_and_check_task_config(
        seq_to_seq_cfg
    )

    with TemporaryDirectory() as tmpdir:
        # Init Saver Callback, and simulate x2 epochs to save state_dict
        cb_handler = CallbackHandler(
            callbacks=[Saver(log_dir=tmpdir, model=seq_to_seq)]
        )

        cb_handler.on_train_begin(epochs)
        cb_handler.on_epoch_end()
        cb_handler.on_epoch_end()  # saves state_dict

        # make copy of saved state_dict
        expected_sd = copy.deepcopy(seq_to_seq.state_dict())

        # Alter state_dict so that it != expected_sd
        change_seq_to_seq_state_dict(seq_to_seq)

        # check that state_dict **has** changed by checking that all subdicts
        # are altered
        with pytest.raises(AssertionError):
            check_state_dicts_match(
                expected_sd["model"], seq_to_seq.state_dict()["model"]
            )
        if seq_to_seq.optim:
            with pytest.raises(AssertionError):
                check_state_dicts_match(
                    expected_sd["optim"], seq_to_seq.state_dict()["optim"]
                )
        if seq_to_seq.lr_scheduler:
            with pytest.raises(AssertionError):
                check_state_dicts_match(
                    expected_sd["lr_scheduler"],
                    seq_to_seq.state_dict()["lr_scheduler"],
                )

        # Now use **newly initialised** Saver cb to reload the state_dict
        cb_handler = CallbackHandler(
            callbacks=[Saver(log_dir=tmpdir, model=seq_to_seq)]
        )

        cb_handler.on_train_begin(epochs)

        check_state_dicts_match(expected_sd, seq_to_seq.state_dict())
        assert cb_handler.state_dict["epoch"] == 2
        assert cb_handler.state_dict["total_train_batches"] == 0

        # It should also be possible to reload state_dict when fit orchestrates
        # the callbacks. First change state_dict again:
        change_seq_to_seq_state_dict(seq_to_seq)

        # checking state_dict has changed is expensive so skip here.

        # 'Run training' for no epochs - the saver callback should re-load
        # seq_to_seq state_dict == expected_sd
        fit(
            seq_to_seq,
            epochs=0,
            train_loader=train_loader,
            eval_loader=None,
            callbacks=[Saver(log_dir=tmpdir, model=seq_to_seq)],
        )

        check_state_dicts_match(expected_sd, seq_to_seq.state_dict())
