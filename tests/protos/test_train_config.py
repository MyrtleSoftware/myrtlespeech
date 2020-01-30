from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from google.protobuf.wrappers_pb2 import FloatValue
from myrtlespeech.protos import train_config_pb2

from tests.protos.test_dataset import datasets
from tests.protos.test_lr_scheduler import constant_lrs
from tests.protos.test_lr_scheduler import cosine_annealing_lrs
from tests.protos.test_lr_scheduler import exponential_lrs
from tests.protos.test_lr_scheduler import step_lrs
from tests.protos.test_optimizer import adams
from tests.protos.test_optimizer import sgds
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def train_configs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[train_config_pb2.TrainConfig],
    st.SearchStrategy[Tuple[train_config_pb2.TrainConfig, Dict]],
]:
    """Returns a SearchStrategy for a TrainConfig + maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["batch_size"] = draw(st.integers(min_value=1, max_value=128))
    kwargs["epochs"] = draw(st.integers(min_value=1, max_value=128))

    # optimizer
    descript = train_config_pb2.TrainConfig.DESCRIPTOR
    optim_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name["supported_optimizers"].fields
            ]
        )
    )
    if optim_str == "sgd":
        kwargs[optim_str] = draw(sgds())
    elif optim_str == "adam":
        kwargs[optim_str] = draw(adams())
    else:
        raise ValueError(f"unknown optim type {optim_str}")

    # learning rate scheduler
    lr_scheduler_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name[
                    "supported_lr_scheduler"
                ].fields
            ]
        )
    )
    if lr_scheduler_str == "constant_lr":
        kwargs[lr_scheduler_str] = draw(constant_lrs())
    elif lr_scheduler_str == "step_lr":
        kwargs[lr_scheduler_str] = draw(step_lrs())
    elif lr_scheduler_str == "exponential_lr":
        kwargs[lr_scheduler_str] = draw(exponential_lrs())
    elif lr_scheduler_str == "cosine_annealing_lr":
        kwargs[lr_scheduler_str] = draw(cosine_annealing_lrs())
    else:
        raise ValueError(f"unknown lr_scheduler type {lr_scheduler_str}")

    kwargs["dataset"] = draw(datasets())

    # shuffle
    shuffle_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name["supported_shuffles"].fields
            ]
        )
    )
    if shuffle_str == "shuffle_batches_before_every_epoch":
        kwargs[shuffle_str] = draw(st.booleans())
    else:
        raise ValueError(f"unknown shuffle type {shuffle_str}")

    # label smoothing
    to_ignore: List = []
    if st.booleans():
        kwargs["label_smoothing"] = draw(_label_smoothings())
    else:
        to_ignore.append("label_smoothing")
    # initialise and return
    all_fields_set(train_config_pb2.TrainConfig, kwargs, to_ignore=to_ignore)
    train_config = train_config_pb2.TrainConfig(**kwargs)
    if not return_kwargs:
        return train_config
    return train_config, kwargs


@st.composite
def smoothing_types(draw) -> st.SearchStrategy:
    """Returns a SearchStrategy for a stage."""
    return draw(
        st.sampled_from(
            train_config_pb2.TrainConfig.LabelSmoothing.SmoothingType.values()
        )
    )


@st.composite
def _label_smoothings(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[train_config_pb2.TrainConfig.LabelSmoothing],
    st.SearchStrategy[
        Tuple[train_config_pb2.TrainConfig.LabelSmoothing, Dict]
    ],
]:
    """Returns a SearchStrategy for label_smoothing + maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["type"] = draw(smoothing_types())

    kwargs["probability"] = FloatValue(value=draw(st.floats(0.0, 0.3)))

    all_fields_set(train_config_pb2.TrainConfig.LabelSmoothing, kwargs)
    label_smoothing = train_config_pb2.TrainConfig.LabelSmoothing(**kwargs)
    if not return_kwargs:
        return label_smoothing
    return label_smoothing, kwargs
