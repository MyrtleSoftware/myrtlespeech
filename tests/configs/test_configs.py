import os
from glob import glob

import google.protobuf.text_format as text_format  # weird import for mypy
import pytest
from myrtlespeech import configs
from myrtlespeech.builders.task_config import build
from myrtlespeech.protos import task_config_pb2

# Utilities -------------------------------------------------------------------

DATASET_STR = "dataset "
FAKE_DSET = """{
fake_speech_to_text {
    dataset_len: 2;
    audio_ms {
        lower: 1;
        upper: 10;
        }
    label_symbols: "abc";
    label_len {
        lower: 1;
        upper: 10;
        }
    }
}"""


def replace_dataset_w_fake_dataset(config):
    """Replaces dataset in string config with fake_speech_to_text.

    It does this by stepping through the config string one character at a time.

    Note: this function looks for the sequence :py:data:`dataset `
    so this test will fail if any other elemtents of the proto (valid or not)
    contain this string.
    """
    dset_count = 0  # Number of times DATASET_STR has been found
    dset_idx = 0  # Current index in DATASET_STR
    reading_dataset = False  # If True, DATASET_STR is currently being read
    config_out = ""  # Config string that will be returned
    for idx, char in enumerate(config):
        if dset_idx == len(DATASET_STR):
            reading_dataset = True
            dset_idx = 0
            bracket_depth = 0  # tracks depth inside brackets
            entered_brackets = False  # ensure brackets are entered to depth
            # of at least one
        if not reading_dataset:
            if char == DATASET_STR[dset_idx]:
                dset_idx += 1
            else:
                dset_idx = 0
            config_out += char  # Add char unchanged
        else:
            if char == "{":
                bracket_depth += 1
                entered_brackets = True
            elif char == "}":
                bracket_depth -= 1
            else:
                # char **not** added whilst reading_dataset = True
                pass

            if bracket_depth == 0 and entered_brackets:
                # ... then we have reached end of dataset config string
                config_out += FAKE_DSET
                dset_count += 1
                reading_dataset = False
    if dset_count != 2:
        raise ValueError(
            f"The string `dataset ` should appear exactly twice in a valid "
            f"config file but it appears {dset_count} times"
        )
    return config_out


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
    """Ensures :py:class:`seq_to_seq` in config file can be built.

    This attempst to build the task config **minus the dataset** which is
    replaced with fake_speech_to_text for speed.
    """
    with open(config_path, "r") as config_file:
        config = config_file.read()

    new_config = replace_dataset_w_fake_dataset(config)
    compiled = text_format.Merge(new_config, task_config_pb2.TaskConfig())
    build(compiled)
