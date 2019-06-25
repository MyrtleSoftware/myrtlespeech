import os
from glob import glob

import pytest
import google.protobuf.text_format as text_format  # weird import for mypy

from myrtlespeech.builders.encoder_decoder_builder import build
from myrtlespeech import configs
from myrtlespeech.protos import encoder_decoder_pb2


# Fixtures and Strategies -----------------------------------------------------


@pytest.fixture(
    params=glob(os.path.join(os.path.dirname(configs.__file__), "*.config"))
)
def config_path(request):
    """Fixture to return all 'myrtlespeech/configs/*.config' files."""
    return request.param


# Tests -----------------------------------------------------------------------


def test_all_configs_build(config_path):
    """Ensures all config files in `myrtlespeech/config/` `build`."""
    with open(config_path, "r") as config_file:
        config = config_file.read()
    enc_dec = text_format.Merge(config, encoder_decoder_pb2.EncoderDecoder())
    build(enc_dec)
