from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import task_config_pb2

from tests.protos.test_eval_config import eval_configs
from tests.protos.test_speech_to_text import speech_to_texts
from tests.protos.test_train_config import train_configs
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def task_configs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[task_config_pb2.TaskConfig],
    st.SearchStrategy[Tuple[task_config_pb2.TaskConfig, Dict]],
]:
    """Returns a SearchStrategy for a TaskConfig plus maybe the kwargs."""
    kwargs: Dict = {}

    descript = task_config_pb2.TaskConfig.DESCRIPTOR

    # model
    model_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["supported_models"].fields]
        )
    )
    if model_str == "speech_to_text":
        kwargs[model_str] = draw(speech_to_texts())
    else:
        raise ValueError(f"unknown model type {model_str}")

    # train config
    kwargs["train_config"] = draw(train_configs())

    # eval config
    kwargs["eval_config"] = draw(eval_configs())

    # initialise and return
    all_fields_set(task_config_pb2.TaskConfig, kwargs)
    task_config = task_config_pb2.TaskConfig(**kwargs)
    if not return_kwargs:
        return task_config
    return task_config, kwargs
