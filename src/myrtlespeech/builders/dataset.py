from typing import Callable, Optional

import torch

from myrtlespeech.data.dataset.fake import FakeDataset, speech_to_text
from myrtlespeech.protos import dataset_pb2


def build(
    dataset: dataset_pb2.Dataset,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    add_seq_len_to_transforms: bool = False,
) -> torch.utils.data.Dataset:
    """Returns a :py:class:`torch.utils.data.Dataset` based on the config.

    Args:
        dataset: A ``Dataset`` protobuf object containing the config for the
            desired :py:class:`torch.utils.data.Dataset`.

        transform: TODO

        target_transform: TODO

        add_seq_len_to_transforms: TODO

    Returns:
        A :py:class:`torch.utils.data.Dataset` based on the config.
    """
    supported_dataset = dataset.WhichOneof("supported_datasets")

    if add_seq_len_to_transforms:
        transform = _add_seq_len(transform)
        target_transform = _add_seq_len(target_transform)

    if supported_dataset == "fake_speech_to_text":
        cfg = dataset.fake_speech_to_text
        dataset = FakeDataset(
            generator=speech_to_text(
                audio_ms=(cfg.audio_ms.lower, cfg.audio_ms.upper),
                label_symbols=cfg.label_symbols,
                label_len=(cfg.label_len.lower, cfg.label_len.upper),
                audio_transform=transform,
                label_transform=target_transform,
            ),
            dataset_len=cfg.dataset_len,
        )
    else:
        raise ValueError(f"{supported_dataset} not supported")

    return dataset


def _add_seq_len(transform: Optional[Callable]) -> Callable:
    def new_transform(x, *args, **kwargs):
        seq_len = torch.tensor(len(x), requires_grad=False)
        result = x
        if transform is not None:
            result = transform(x, *args, **kwargs)
        return result, seq_len

    return new_transform
