import torch

from myrtlespeech.data.dataset.fake import FakeDataset, speech_to_text
from myrtlespeech.protos import dataset_pb2


def build(dataset: dataset_pb2.Dataset) -> torch.utils.data.Dataset:
    """Returns a :py:class:`torch.utils.data.Dataset` based on the config.

    Args:
        dataset: A ``Dataset`` protobuf object containing the config for the
            desired :py:class:`torch.utils.data.Dataset`.

    Returns:
        A :py:class:`torch.utils.data.Dataset` based on the config.
    """
    supported_dataset = dataset.WhichOneof("supported_datasets")

    if supported_dataset == "fake_speech_to_text":
        cfg = dataset.fake_speech_to_text
        dataset = FakeDataset(
            generator=speech_to_text(
                audio_ms=(cfg.audio_ms.lower, cfg.audio_ms.upper),
                label_symbols=cfg.label_symbols,
                label_len=(cfg.label_len.lower, cfg.label_len.upper),
            ),
            dataset_len=cfg.dataset_len,
        )
    else:
        raise ValueError(f"{supported_dataset} not supported")

    return dataset
