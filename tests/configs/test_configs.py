import os
from glob import glob

import google.protobuf.text_format as text_format  # weird import for mypy
import pytest
from myrtlespeech import configs
from myrtlespeech.builders.task_config import build
from myrtlespeech.protos import task_config_pb2

# Utilities -------------------------------------------------------------------


def replace_dataset_w_fake_dataset(config):
    """Replaces dataset proto config with fake_speech_to_text."""
    config.fake_speech_to_text.dataset_len = 2
    config.fake_speech_to_text.audio_ms.lower = 1
    config.fake_speech_to_text.audio_ms.upper = 10
    config.fake_speech_to_text.label_symbols = "abc"
    config.fake_speech_to_text.label_len.lower = 1
    config.fake_speech_to_text.label_len.upper = 10


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


def test_model_in_configs_can_be_built(config_path):
    """Ensures :py:class:`task_config` in .config file can be built.

    This attempts to build the task config **minus the dataset** which is
    replaced with fake_speech_to_text for speed.
    """
    with open(config_path, "r") as config_file:
        config = config_file.read()

    compiled = text_format.Merge(config, task_config_pb2.TaskConfig())
    replace_dataset_w_fake_dataset(compiled.train_config.dataset)
    replace_dataset_w_fake_dataset(compiled.eval_config.dataset)
    build(compiled)
