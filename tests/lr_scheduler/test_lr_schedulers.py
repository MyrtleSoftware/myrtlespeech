import copy
from typing import Dict
from typing import Optional
from typing import Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.lr_scheduler.base import LRSchedulerBase
from myrtlespeech.lr_scheduler.constant import ConstantLR
from myrtlespeech.lr_scheduler.poly import PolynomialLR

from tests.lr_scheduler.utils import check_lr_schedule_set
from tests.utils.utils import check_state_dicts_match

# Fixtures and Strategies -----------------------------------------------------

TOL = 1e-8


def lr_scheduler_class() -> st.SearchStrategy[type]:
    """Returns a strategy which generates numpy variant of torch data types."""
    return st.sampled_from(
        [
            ConstantLR,
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.ExponentialLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
            PolynomialLR,
        ]
    )


@st.composite
def _lr_scheduler_data(
    draw, cls: Optional[type] = None, warmup: bool = False,
) -> st.SearchStrategy[
    Tuple[torch.optim.lr_scheduler._LRScheduler, Dict],
]:
    """Returns a search strategy for lr_scheduler.
    """
    batches_per_epoch = draw(st.integers(min_value=1, max_value=8))
    epochs = draw(st.integers(min_value=4, max_value=30))
    initial_lr = draw(st.floats(min_value=1e-5, max_value=1.0))
    num_warmup_steps = None
    if warmup:
        num_warmup_steps = draw(st.integers(min_value=2, max_value=10))
    if cls is None:
        cls = draw(lr_scheduler_class())
    kwargs = {}
    if cls == ConstantLR:
        step_freq = 1  # This could be any value
    elif cls == torch.optim.lr_scheduler.StepLR:
        kwargs["gamma"] = draw(st.floats(min_value=0.1, max_value=0.999))
        kwargs["step_size"] = draw(st.integers(1, max_value=20))
        step_freq = batches_per_epoch
    elif cls == torch.optim.lr_scheduler.ExponentialLR:
        kwargs["gamma"] = draw(st.floats(min_value=0.1, max_value=0.999))
        step_freq = batches_per_epoch
    elif cls == torch.optim.lr_scheduler.CosineAnnealingLR:
        kwargs["eta_min"] = draw(st.floats(min_value=1e-7, max_value=1.0))
        kwargs["T_max"] = draw(st.integers(1, max_value=20))
        step_freq = batches_per_epoch
    elif cls == PolynomialLR:
        kwargs["total_steps"] = batches_per_epoch * epochs
        kwargs["min_lr_multiple"] = draw(
            st.floats(min_value=1e-5, max_value=0.1)
        )
        step_freq = 1
    else:
        raise ValueError(
            f"cls={cls} is not a recognized class of lr_scheduler"
        )

    optimizer = torch.optim.SGD(
        torch.nn.Linear(2, 3).parameters(), lr=initial_lr
    )

    scheduler = LRSchedulerBase(
        scheduler=cls(optimizer=optimizer, **kwargs),
        scheduler_step_freq=step_freq,
        num_warmup_steps=num_warmup_steps,
    )
    params = {
        "batches_per_epoch": batches_per_epoch,
        "epochs": epochs,
        "initial_lr": initial_lr,
        "num_warmup_steps": num_warmup_steps,
        "min_lr_multiple": kwargs.get("min_lr_multiple"),
    }
    return scheduler, params


# Tests -----------------------------------------------------------------------


@given(lr_scheduler_data=_lr_scheduler_data(),)
def test_lr_schedule_set_correctly_no_warmup(
    lr_scheduler_data: Tuple[torch.optim.lr_scheduler._LRScheduler, Dict]
) -> None:
    """Ensures lr_scheduler correctly sets optimizer lrs w/o warmup."""
    scheduler, kwargs = lr_scheduler_data
    batches_per_epoch = kwargs["batches_per_epoch"]
    epochs = kwargs["batches_per_epoch"]
    initial_lr = kwargs["initial_lr"]
    min_lr_multiple = kwargs.get("min_lr_multiple")

    results = check_lr_schedule_set(scheduler, batches_per_epoch, epochs)

    # Check first lr is equal to initial learning rate
    assert abs(results[0]["lr"] - initial_lr) < TOL

    # Check lr changes with desired frequency
    if scheduler._scheduler.__class__ in [
        torch.optim.lr_scheduler.StepLR,
        torch.optim.lr_scheduler.ExponentialLR,
        torch.optim.lr_scheduler.CosineAnnealingLR,
    ]:
        expected_freq = batches_per_epoch
    elif scheduler._scheduler.__class__ in [PolynomialLR]:
        expected_freq = 1
    elif scheduler._scheduler.__class__ == ConstantLR:
        return
    else:
        raise ValueError(
            f"unknown scheduler._scheduler.__class__="
            f"{scheduler._scheduler.__class__}"
        )

    prev_lr = -1.0
    for step, data in results.items():
        lr_changed = abs(data["lr"] - prev_lr) > TOL
        if lr_changed:
            assert step % expected_freq == 0
        prev_lr = data["lr"]

        if scheduler._scheduler.__class__ == PolynomialLR:
            assert data["lr"] + TOL > min_lr_multiple * initial_lr


@given(lr_scheduler_data=_lr_scheduler_data(warmup=True),)
def test_lr_schedule_set_correctly_with_warmup(
    lr_scheduler_data: Tuple[torch.optim.lr_scheduler._LRScheduler, Dict]
) -> None:
    """Ensures lr_scheduler correctly sets optimizer lrs with warmup."""
    scheduler, kwargs = lr_scheduler_data
    batches_per_epoch = kwargs["batches_per_epoch"]
    epochs = kwargs["batches_per_epoch"]
    num_warmup_steps = kwargs["num_warmup_steps"]

    results = check_lr_schedule_set(scheduler, batches_per_epoch, epochs)
    assert scheduler.num_warmup_steps == num_warmup_steps
    # Check first lr is equal to zero (i.e. using warmup)
    assert abs(results[0]["lr"] - 0.0) < TOL

    # Check lr changes with desired frequency
    if scheduler._scheduler.__class__ in [
        torch.optim.lr_scheduler.StepLR,
        torch.optim.lr_scheduler.ExponentialLR,
        torch.optim.lr_scheduler.CosineAnnealingLR,
    ]:
        expected_freq = batches_per_epoch
    elif scheduler._scheduler.__class__ in [PolynomialLR]:
        expected_freq = 1
    elif scheduler._scheduler.__class__ == ConstantLR:
        expected_freq = -1
    else:
        raise ValueError(
            f"unknown scheduler._scheduler.__class__="
            f"{scheduler._scheduler.__class__}"
        )

    prev_lr = -1.0
    for step, data in results.items():
        lr_changed = abs(data["lr"] - prev_lr) > TOL
        if step <= num_warmup_steps:
            assert lr_changed
        elif lr_changed:
            assert step % expected_freq == 0
        prev_lr = data["lr"]


@given(data=st.data(), warmup=st.booleans())
def test_lr_schedule_state_dict_restores_correctly(data, warmup,) -> None:
    """Ensures lr_scheduler state dict is correctly restored."""

    scheduler, _ = data.draw(_lr_scheduler_data(warmup=warmup))
    old_state_dict = copy.deepcopy(scheduler.state_dict())

    scheduler2, _ = data.draw(
        _lr_scheduler_data(cls=scheduler._scheduler.__class__, warmup=warmup)
    )
    scheduler2.load_state_dict(old_state_dict)
    new_state_dict = scheduler2.state_dict()

    check_state_dicts_match(old_state_dict, new_state_dict)
