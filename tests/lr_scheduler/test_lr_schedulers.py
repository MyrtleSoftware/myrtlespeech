import copy
from typing import Optional
from typing import Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.lr_scheduler.base import LRSchedulerBase
from myrtlespeech.lr_scheduler.constant import ConstantLR
from myrtlespeech.lr_scheduler.poly import PolynomialLR

from tests.lr_scheduler.utils import check_lr_schedule_set


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
    draw, cls: Optional[type] = None,
) -> st.SearchStrategy[
    Tuple[torch.optim.lr_scheduler._LRScheduler, int, int, int],
]:
    """Returns a search strategy for lr_scheduler.
    """
    batches_per_epoch = draw(st.integers(min_value=1, max_value=8))
    epochs = draw(st.integers(min_value=4, max_value=30))
    initial_lr = draw(st.floats(min_value=1e-5, max_value=1.0))
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
        num_warmup_steps=None,
    )
    return (
        scheduler,
        batches_per_epoch,
        epochs,
        initial_lr,
    )


# Utilities -------------------------------------------------------------------


def check_state_dicts_match(dict1, dict2):
    """Ensures state_dicts match."""
    assert dict1.keys() == dict2.keys()
    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, float) or isinstance(val2, float):
            assert abs(val1 - val2) < TOL
        else:
            assert val1 == val2


# Tests -----------------------------------------------------------------------


@given(lr_scheduler_data=_lr_scheduler_data(),)
def test_lr_schedule_set_correctly_and_first_lr_correct(
    lr_scheduler_data: Tuple[
        torch.optim.lr_scheduler._LRScheduler, int, int, int
    ]
) -> None:
    """Ensures lr_scheduler correctly sets optimizer lrs."""
    scheduler, batches_per_epoch, epochs, initial_lr = lr_scheduler_data

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


@given(data=st.data(), lr_scheduler_data=_lr_scheduler_data())
def test_lr_schedule_state_dict_restores_correctly(
    data,
    lr_scheduler_data: Tuple[
        torch.optim.lr_scheduler._LRScheduler, int, int, int
    ],
) -> None:
    """Ensures lr_scheduler state dict is correctly restored."""
    scheduler, _, _, _ = lr_scheduler_data

    old_state_dict = copy.deepcopy(scheduler.state_dict())
    scheduler2, _, _, _ = data.draw(
        _lr_scheduler_data(cls=scheduler._scheduler.__class__)
    )
    scheduler2.load_state_dict(old_state_dict)
    new_state_dict = scheduler2.state_dict()

    check_state_dicts_match(old_state_dict, new_state_dict)
