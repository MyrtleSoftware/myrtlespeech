from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import shuffle_strategy_pb2
from myrtlespeech.protos import train_config_pb2

from tests.protos.test_dataset import datasets
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

    kwargs["dataset"] = draw(datasets())

    # shuffle
    shuffle_str = draw(
        st.sampled_from(
            [
                f.name
                for f in descript.oneofs_by_name["shuffle_strategy"].fields
            ]
        )
    )
    if shuffle_str == "sequential_batches":
        kwargs[shuffle_str] = shuffle_strategy_pb2.SequentialBatches()
    elif shuffle_str == "random_batches":
        kwargs[shuffle_str] = shuffle_strategy_pb2.RandomBatches()
    elif shuffle_str == "sortagrad":
        kwargs[shuffle_str] = shuffle_strategy_pb2.SortaGrad()
    else:
        raise ValueError(f"unknown shuffle strategy type {shuffle_str}")

    # initialise and return
    all_fields_set(train_config_pb2.TrainConfig, kwargs)
    train_config = train_config_pb2.TrainConfig(**kwargs)
    if not return_kwargs:
        return train_config
    return train_config, kwargs
