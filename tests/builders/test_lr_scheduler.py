from typing import Optional

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from myrtlespeech.builders.lr_scheduler import build
from myrtlespeech.lr_scheduler.constant import ConstantLR
from myrtlespeech.lr_scheduler.poly import PolynomialLR
from myrtlespeech.lr_scheduler.warmup import _LRSchedulerWarmup
from myrtlespeech.protos import train_config_pb2

from tests.protos.test_train_config import train_configs


# Utilities -------------------------------------------------------------------


def lr_scheduler_match_cfg(
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_config: train_config_pb2.TrainConfig,
    batches_per_epoch: int,
    epochs: int,
) -> None:
    """Ensures the Dataset matches protobuf configuration."""
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
    assert isinstance(lr_scheduler.optimizer, torch.optim.Optimizer)

    step_freq: Optional[int] = None
    if isinstance(lr_scheduler, _LRSchedulerWarmup):
        assert train_config.HasField("lr_warmup")
        assert (
            train_config.lr_warmup.num_warmup_steps
            == lr_scheduler.num_warmup_steps
        )
        step_freq = lr_scheduler.step_freq
        lr_scheduler = lr_scheduler._scheduler

    lr_scheduler_str = train_config.WhichOneof("supported_lr_scheduler")
    if isinstance(lr_scheduler, ConstantLR):
        assert lr_scheduler_str == "constant_lr"
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
        assert lr_scheduler_str == "step_lr"
        cfg = train_config.step_lr
        if cfg.HasField("gamma"):
            assert lr_scheduler.gamma == cfg.gamma.value
        assert lr_scheduler.step_size == cfg.step_size
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR):
        assert lr_scheduler_str == "exponential_lr"
        cfg = train_config.exponential_lr
        assert lr_scheduler.gamma == cfg.gamma
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        assert lr_scheduler_str == "cosine_annealing_lr"
        cfg = train_config.cosine_annealing_lr
        if cfg.HasField("eta_min"):
            assert lr_scheduler.eta_min == cfg.eta_min.value
        assert lr_scheduler.T_max == cfg.t_max
    elif isinstance(lr_scheduler, PolynomialLR):
        assert lr_scheduler_str == "polynomial_lr"
        cfg = train_config.polynomial_lr
        assert lr_scheduler.total_steps == batches_per_epoch * epochs
        assert hasattr(lr_scheduler, "min_lr_multiple")
    else:
        raise ValueError(
            f"lr_scheduler_str={lr_scheduler_str} does not match"
            f"lr_scheduler.__class__.__name__ = "
            f"{lr_scheduler.__class__.__name__}"
        )

    # check step_freq
    if step_freq is not None:
        if lr_scheduler_str in [
            "step_lr",
            "exponential_lr",
            "cosine_annealing_lr",
        ]:
            assert step_freq == batches_per_epoch  # Step at end of each epoch
        elif lr_scheduler_str in ["polynomial_lr"]:
            assert step_freq == 1


# Tests -----------------------------------------------------------------------


@given(
    train_config=train_configs(),
    batches_per_epoch=st.integers(1, 200),
    epochs=st.integers(1, 5),
)
def test_build_lr_scheduler_returns_correct_lr_scheduler(
    train_config: train_config_pb2.TrainConfig,
    batches_per_epoch: int,
    epochs: int,
) -> None:
    """Ensures lr_scheduler returned by ``build`` has correct structure."""
    # init optimizer
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 3).parameters(), lr=1.0)

    # build lr_scheduler
    lr_scheduler = build(
        train_config=train_config,
        optimizer=optimizer,
        batches_per_epoch=batches_per_epoch,
        epochs=epochs,
    )
    lr_scheduler_match_cfg(
        lr_scheduler, train_config, batches_per_epoch, epochs,
    )


@given(train_config=train_configs())
def test_unknown_lr_scheduler_raises_value_error(
    train_config: train_config_pb2.TrainConfig,
) -> None:
    """Ensures ValueError is raised when lr_scheduler is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    train_config.ClearField(train_config.WhichOneof("supported_lr_scheduler"))
    if train_config.HasField("lr_warmup"):
        train_config.ClearField("lr_warmup")
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 3).parameters(), lr=1.0)

    with pytest.raises(ValueError):
        build(
            train_config=train_config,
            optimizer=optimizer,
            batches_per_epoch=2,
            epochs=5,
        )
