import pytest
import torch
from hypothesis import given

from myrtlespeech.builders.dataset import build
from myrtlespeech.data.dataset.fake import FakeDataset
from myrtlespeech.protos import dataset_pb2
from tests.protos.test_dataset import datasets


# Utilities -------------------------------------------------------------------


def dataset_match_cfg(
    dataset: torch.utils.data.Dataset, dataset_cfg: dataset_pb2.Dataset
) -> None:
    """Ensures the Dataset matches protobuf configuration."""
    assert isinstance(dataset, torch.utils.data.Dataset)

    if isinstance(dataset, FakeDataset) and dataset_cfg.HasField(
        "fake_speech_to_text"
    ):
        cfg = dataset_cfg.fake_speech_to_text

        assert len(dataset) == cfg.dataset_len

        for audio, label in dataset:
            sample_rate = 16000 / 1000  # 16 kHz, convert to ms
            assert cfg.audio_ms.lower * sample_rate <= len(audio)
            assert len(audio) <= cfg.audio_ms.upper * sample_rate
            assert all([symbol in cfg.label_symbols for symbol in label])
            assert cfg.label_len.lower <= len(label) <= cfg.label_len.upper
    else:
        raise ValueError("invalid dataset={dataset}, dataset_cfg={dataset_cfg}")


# Tests -----------------------------------------------------------------------


@given(dataset_cfg=datasets())
def test_build_dataset_returns_correct_dataset(
    dataset_cfg: dataset_pb2.Dataset,
) -> None:
    """Ensures Dataset returned by ``build`` has correct structure."""
    dataset = build(dataset_cfg)
    dataset_match_cfg(dataset, dataset_cfg)


@given(dataset_cfg=datasets())
def test_unknown_dataset_raises_value_error(
    dataset_cfg: dataset_pb2.Dataset
) -> None:
    """Ensures ValueError is raised when dataset is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    dataset_cfg.ClearField(dataset_cfg.WhichOneof("supported_datasets"))
    with pytest.raises(ValueError):
        build(dataset_cfg)
