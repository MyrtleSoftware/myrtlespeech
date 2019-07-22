import os
from glob import glob

import pytest
import google.protobuf.text_format as text_format  # weird import for mypy

from myrtlespeech import configs
from myrtlespeech.protos import task_config_pb2


# Fixtures and Strategies -----------------------------------------------------


@pytest.fixture(
    params=glob(os.path.join(os.path.dirname(configs.__file__), "*.config"))
)
def config_path(request):
    """Fixture to return all 'myrtlespeech/configs/*.config' files."""
    return request.param


# Tests -----------------------------------------------------------------------


def test_all_configs_build(config_path):
    """Ensures all `myrtlespeech/config/*.config` files parse."""
    with open(config_path, "r") as config_file:
        config = config_file.read()
    text_format.Merge(config, task_config_pb2.TaskConfig())
