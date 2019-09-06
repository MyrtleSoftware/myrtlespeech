from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import eval_config_pb2

from tests.protos.test_dataset import datasets
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def eval_configs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[eval_config_pb2.EvalConfig],
    st.SearchStrategy[Tuple[eval_config_pb2.EvalConfig, Dict]],
]:
    """Returns a SearchStrategy for a EvalConfig plus maybe the kwargs."""
    kwargs: Dict = {}

    kwargs["batch_size"] = draw(st.integers(min_value=1, max_value=128))
    kwargs["dataset"] = draw(datasets())

    # initialise and return
    all_fields_set(eval_config_pb2.EvalConfig, kwargs)
    eval_config = eval_config_pb2.EvalConfig(**kwargs)
    if not return_kwargs:
        return eval_config
    return eval_config, kwargs
