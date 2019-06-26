from typing import Union, Tuple, Dict

import hypothesis.strategies as st
from google.protobuf import empty_pb2

from myrtlespeech.protos import encoder_pb2
from myrtlespeech.protos import rnn_pb2
from myrtlespeech.protos import vgg_pb2
from tests.protos import test_rnn
from tests.protos import test_vgg
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def encoders(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[encoder_pb2.Encoder],
    st.SearchStrategy[Tuple[encoder_pb2.Encoder, Dict]],
]:
    """Returns a SearchStrategy for Encoder plus maybe the kwargs."""
    kwargs: Dict = {}

    # initialise oneof supported_cnns
    cnn = draw(_supported_cnns())
    if isinstance(cnn, empty_pb2.Empty):
        kwargs["no_cnn"] = cnn
    elif isinstance(cnn, vgg_pb2.VGG):
        kwargs["vgg"] = cnn
    else:
        raise ValueError(f"unknown cnn type {type(cnn)}")

    # initialise oneof supported_rnns
    rnn = draw(_supported_rnns())
    if isinstance(rnn, empty_pb2.Empty):
        kwargs["no_rnn"] = rnn
    elif isinstance(rnn, rnn_pb2.RNN):
        kwargs["rnn"] = rnn
    else:
        raise ValueError(f"unknown rnn type {type(rnn)}")

    # initialise encoder and return
    all_fields_set(encoder_pb2.Encoder, kwargs)
    encoder = encoder_pb2.Encoder(**kwargs)
    if not return_kwargs:
        return encoder
    return encoder, kwargs


@st.composite
def _supported_cnns(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[empty_pb2.Empty],
    st.SearchStrategy[Tuple[empty_pb2.Empty, Dict]],
    st.SearchStrategy[vgg_pb2.VGG],
    st.SearchStrategy[Tuple[vgg_pb2.VGG, Dict]],
]:
    """Returns a SearchStrategy for supported_cnns plus maybe the kwargs."""
    kwargs: Dict = {}

    # verify test can generate all "supported_cnns" and draw one
    supported_cnn_fields = set()
    descriptor = encoder_pb2.Encoder.DESCRIPTOR
    for f in descriptor.oneofs_by_name["supported_cnns"].fields:
        supported_cnn_fields.add(f.name)
    if supported_cnn_fields != {"no_cnn", "vgg"}:
        raise ValueError("update tests to support all supported_cnns")
    cnn_type = draw(st.sampled_from([empty_pb2.Empty, vgg_pb2.VGG]))

    # get kwargs for chosen cnn_type
    if cnn_type == empty_pb2.Empty:
        pass
    elif cnn_type == vgg_pb2.VGG:
        _, kwargs = draw(test_vgg.vggs(return_kwargs=True))

    # initialise cnn_type and return
    all_fields_set(cnn_type, kwargs)
    cnn = cnn_type(**kwargs)
    if not return_kwargs:
        return cnn
    return cnn, kwargs


@st.composite
def _supported_rnns(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[empty_pb2.Empty],
    st.SearchStrategy[Tuple[empty_pb2.Empty, Dict]],
    st.SearchStrategy[rnn_pb2.RNN],
    st.SearchStrategy[Tuple[rnn_pb2.RNN, Dict]],
]:
    """Returns a SearchStrategy for supported_rnns plus maybe the kwargs."""
    kwargs: Dict = {}

    # verify test can generate all "supported_rnns" and draw one
    supported_rnn_fields = set()
    descriptor = encoder_pb2.Encoder.DESCRIPTOR
    for f in descriptor.oneofs_by_name["supported_rnns"].fields:
        supported_rnn_fields.add(f.name)
    if supported_rnn_fields != {"no_rnn", "rnn"}:
        raise ValueError("update tests to support all supported_rnns")
    rnn_type = draw(st.sampled_from([empty_pb2.Empty, rnn_pb2.RNN]))

    # get kwargs for chosen rnn_type
    if rnn_type == empty_pb2.Empty:
        pass
    elif rnn_type == rnn_pb2.RNN:
        _, kwargs = draw(test_rnn.rnns(return_kwargs=True))

    # initialise rnn_type and return
    all_fields_set(rnn_type, kwargs)
    rnn = rnn_type(**kwargs)
    if not return_kwargs:
        return rnn
    return rnn, kwargs
