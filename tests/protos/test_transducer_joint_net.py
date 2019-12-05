from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from myrtlespeech.protos import transducer_joint_net_pb2

from tests.protos.test_fully_connected import fully_connecteds
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def transducer_joint_net(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[transducer_joint_net_pb2.TransducerJointNet],
    st.SearchStrategy[
        Tuple[transducer_joint_net_pb2.TransducerJointNet, Dict]
    ],
]:
    """Returns a SearchStrategy for joint net plus maybe the kwargs."""
    kwargs: Dict = {}
    kwargs["fc"] = draw(fully_connecteds(valid_only=True))
    all_fields_set(transducer_joint_net_pb2.TransducerJointNet, kwargs)

    joint_net = transducer_joint_net_pb2.TransducerJointNet(**kwargs)
    if not return_kwargs:
        return joint_net
    return joint_net, kwargs
