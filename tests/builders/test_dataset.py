import google.protobuf.text_format as text_format  # from _ import _ errors
import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from myrtlespeech.builders.dataset import build
from myrtlespeech.data.dataset.fake import FakeDataset
from myrtlespeech.protos import dataset_pb2

from tests.protos.test_dataset import datasets


# Utilities -------------------------------------------------------------------


def dataset_match_cfg(
    dataset: torch.utils.data.Dataset,
    dataset_cfg: dataset_pb2.Dataset,
    add_seq_len_to_transforms: bool,
) -> None:
    """Ensures the Dataset matches protobuf configuration."""
    assert isinstance(dataset, torch.utils.data.Dataset)

    if isinstance(dataset, FakeDataset) and dataset_cfg.HasField(
        "fake_speech_to_text"
    ):
        cfg = dataset_cfg.fake_speech_to_text

        assert len(dataset) == cfg.dataset_len

        sample_rate = 16000 / 1000  # convert 16 kHz to ms for len(audio) test
        for audio, label in dataset:
            if add_seq_len_to_transforms:
                audio, audio_len = audio
                label, label_len = label
                assert len(audio) == audio_len
                assert len(label) == label_len
            assert cfg.audio_ms.lower * sample_rate <= len(audio)
            assert len(audio) <= cfg.audio_ms.upper * sample_rate
            assert all([symbol in cfg.label_symbols for symbol in label])
            assert cfg.label_len.lower <= len(label) <= cfg.label_len.upper
    else:
        raise ValueError(
            "invalid dataset={dataset}, dataset_cfg={dataset_cfg}"
        )


# Tests -----------------------------------------------------------------------


@given(add_seq_len_to_transforms=st.booleans())
def test_build_passes_transform_to_fake_speech_to_text(
    add_seq_len_to_transforms: bool,
) -> None:
    """Unit test to ensure build passes transforms to fake_speech_to_text."""
    dataset_cfg = text_format.Merge(
        """
    fake_speech_to_text {
      dataset_len: 10;
      audio_ms {
        lower: 100;
        upper: 10000;
      }
      label_symbols: "abcde";
      label_len {
        lower: 100;
        upper: 10000;
      }
    }
    """,
        dataset_pb2.Dataset(),
    )
    transform = lambda x: torch.tensor([1.0, 2.0])  # noqa: E731
    target_transform = lambda x: "target transform"  # noqa: E731

    dataset = build(
        dataset=dataset_cfg,
        pre_load_transform=None,
        post_load_transform=transform,
        target_transform=target_transform,
        add_seq_len_to_transforms=add_seq_len_to_transforms,
    )

    for audio, label in dataset:
        if add_seq_len_to_transforms:
            audio, audio_len = audio
            label, label_len = label
            assert len(audio) == audio_len
            assert len(label) == label_len
        assert torch.all(audio == torch.tensor([1.0, 2.0]))
        assert label == "target transform"


@given(dataset_cfg=datasets(), add_seq_len_to_transforms=st.booleans())
def test_build_dataset_returns_correct_dataset(
    dataset_cfg: dataset_pb2.Dataset, add_seq_len_to_transforms: bool
) -> None:
    """Ensures Dataset returned by ``build`` has correct structure."""
    dataset = build(
        dataset=dataset_cfg,
        add_seq_len_to_transforms=add_seq_len_to_transforms,
    )
    dataset_match_cfg(dataset, dataset_cfg, add_seq_len_to_transforms)


@given(dataset_cfg=datasets())
def test_unknown_dataset_raises_value_error(
    dataset_cfg: dataset_pb2.Dataset,
) -> None:
    """Ensures ValueError is raised when dataset is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    dataset_cfg.ClearField(dataset_cfg.WhichOneof("supported_datasets"))
    with pytest.raises(ValueError):
        build(dataset_cfg)
