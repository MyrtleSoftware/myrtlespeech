from typing import Union, Tuple, Dict

import hypothesis.strategies as st
from google.protobuf import empty_pb2

from myrtlespeech.protos import encoder_pb2
from myrtlespeech.protos import rnn_pb2
from myrtlespeech.protos import vgg_pb2
from tests.protos import test_vgg
from tests.protos.test_rnn import rnns
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

    # verify test can generate all "supported_encoders" and draw one
    des = encoder_pb2.Encoder.DESCRIPTOR
    encoder_type_str = draw(
        st.sampled_from(
            [f.name for f in des.oneofs_by_name["supported_encoders"].fields]
        )
    )

    # get kwargs for chosen encoder_type
    if encoder_type_str == "cnn_rnn_encoder":
        kwargs["cnn_rnn_encoder"] = draw(cnn_rnn_encoders())
    else:
        raise ValueError(
            f"test does not support generation of {encoder_type_str}"
        )

    # initialise encoder_type and return
    all_fields_set(encoder_pb2.Encoder, kwargs)
    encoder = encoder_pb2.Encoder(**kwargs)  # type: ignore
    if not return_kwargs:
        return encoder
    return encoder, kwargs


@st.composite
def cnn_rnn_encoders(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[encoder_pb2.Encoder.CNNRNNEncoder],
    st.SearchStrategy[Tuple[encoder_pb2.Encoder.CNNRNNEncoder, Dict]],
]:
    """Returns a SearchStrategy for CNNRNNEncoders plus maybe the kwargs."""
    kwargs = {}

    # initialise oneof supported_cnns
    cnn = draw(_supported_cnns())
    if isinstance(cnn, empty_pb2.Empty):
        kwargs["no_cnn"] = cnn
    elif isinstance(cnn, vgg_pb2.VGG):
        kwargs["vgg"] = cnn
    else:
        raise ValueError(f"unknown cnn type {type(cnn)}")

    # initialise oneof supported_rnns
    rnn = draw(rnns())
    if isinstance(rnn, empty_pb2.Empty):
        kwargs["no_rnn"] = rnn
    elif isinstance(rnn, rnn_pb2.RNN):
        kwargs["rnn"] = rnn
    else:
        raise ValueError(f"unknown rnn type {type(rnn)}")

    # initialise CNNRNNEncoder and return
    all_fields_set(encoder_pb2.Encoder.CNNRNNEncoder, kwargs)
    cnn_rnn_encoder = encoder_pb2.Encoder.CNNRNNEncoder(**kwargs)
    if not return_kwargs:
        return cnn_rnn_encoder
    return cnn_rnn_encoder, kwargs


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
    descript = encoder_pb2.Encoder.CNNRNNEncoder.DESCRIPTOR
    cnn_type_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["supported_cnns"].fields]
        )
    )

    # get kwargs for chosen cnn_type
    if cnn_type_str == "no_cnn":
        cnn_type = empty_pb2.Empty
    elif cnn_type_str == "vgg":
        cnn_type = vgg_pb2.VGG  # type: ignore
        _, kwargs = draw(test_vgg.vggs(return_kwargs=True))
    else:
        raise ValueError(f"test does not support generation of {cnn_type_str}")

    # initialise cnn_type and return
    all_fields_set(cnn_type, kwargs)
    cnn = cnn_type(**kwargs)  # type: ignore
    if not return_kwargs:
        return cnn
    return cnn, kwargs
