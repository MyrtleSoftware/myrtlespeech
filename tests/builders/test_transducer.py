import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.builders.transducer import build as build_transducer
from myrtlespeech.model.transducer import Transducer
from myrtlespeech.protos import transducer_pb2

from tests.protos.test_transducer import transducers


# Tests -----------------------------------------------------------------------


@given(
    transducer_cfg=transducers(),
    input_features=st.integers(min_value=2, max_value=16),
    input_channels=st.integers(min_value=1, max_value=5),
    vocab_size=st.integers(min_value=1, max_value=8),
)
def test_build_transducer(
    transducer_cfg: transducer_pb2.Transducer,
    input_features: int,
    input_channels: int,
    vocab_size: int,
) -> None:
    """Ensures builder does not throw exception."""
    transducer = build_transducer(
        transducer_cfg, input_features, input_channels, vocab_size
    )
    assert isinstance(transducer, torch.nn.Module)
    assert isinstance(transducer, Transducer)
    assert hasattr(transducer, "encode")
    assert isinstance(transducer.encode, torch.nn.Module)
    assert hasattr(transducer, "predict_net")
    assert isinstance(transducer.predict_net, torch.nn.Module)
    assert hasattr(transducer, "joint_net")
    assert isinstance(transducer.joint_net, torch.nn.Module)
