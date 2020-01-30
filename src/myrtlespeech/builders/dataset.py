from typing import Callable
from typing import Optional

import torch
from myrtlespeech.data.dataset.fake import FakeDataset
from myrtlespeech.data.dataset.fake import speech_to_text
from myrtlespeech.data.dataset.librispeech import LibriSpeech
from myrtlespeech.protos import dataset_pb2


def build(
    dataset: dataset_pb2.Dataset,
    pre_load_transform: Optional[Callable] = None,
    post_load_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    add_seq_len_to_transforms: bool = False,
    download: bool = False,
) -> torch.utils.data.Dataset:
    """Returns a :py:class:`torch.utils.data.Dataset` based on the config.

    Args:
        dataset: A :py:class:`myrtlespeech.protos.dataset_pb2.Dataset` protobuf
            object containing the config for the desired
            :py:class:`torch.utils.data.Dataset`.

        pre_load_transform: Pre-loading transforms to pass to the
            :py:class:`torch.utils.data.Dataset`.

        post_load_transform: Transform to pass to the
            :py:class:`torch.utils.data.Dataset`.

        target_transform: Target transform to pass to the
            :py:class:`torch.utils.data.Dataset`.

        add_seq_len_to_transforms: If :py:data:`True`, an additional function
            is applied after ``post_load_transform`` and ``target_transform``
            that takes a value and returns a tuple of ``(value,
            torch.tensor(len(value)))``.

        download: If :py:data:`True` and dataset does not exist, download it
            if possible.

    Returns:
        A :py:class:`torch.utils.data.Dataset` based on the config.

    Example:
        >>> from google.protobuf import text_format
        >>> dataset_cfg = text_format.Merge('''
        ... fake_speech_to_text {
        ...   dataset_len: 2;
        ...   audio_ms {
        ...     lower: 10;
        ...     upper: 100;
        ...   }
        ...   label_symbols: "abcde";
        ...   label_len {
        ...     lower: 1;
        ...     upper: 10;
        ...   }
        ... }
        ... ''', dataset_pb2.Dataset())
        >>> dataset = build(dataset_cfg, add_seq_len_to_transforms=True)
        >>> len(dataset)
        2
        >>> (audio, audio_len), (label, label_len) = dataset[0]
        >>> type(audio)
        <class 'torch.Tensor'>
        >>> bool(audio.size(-1) == audio_len)
        True
        >>> type(label)
        <class 'str'>
        >>> bool(len(label) == label_len)
        True
    """
    supported_dataset = dataset.WhichOneof("supported_datasets")

    if add_seq_len_to_transforms:
        post_load_transform = _add_seq_len(
            post_load_transform, len_fn=lambda x: x.size(-1)
        )
        target_transform = _add_seq_len(target_transform, len_fn=len)

    if supported_dataset == "fake_speech_to_text":
        if pre_load_transform is not None:
            raise ValueError(
                f"Cannot apply pre_load_transform(s) to "
                f"fake_speech_to_text as it is a generator!"
            )
        cfg = dataset.fake_speech_to_text
        dataset = FakeDataset(
            generator=speech_to_text(
                audio_ms=(cfg.audio_ms.lower, cfg.audio_ms.upper),
                label_symbols=cfg.label_symbols,
                label_len=(cfg.label_len.lower, cfg.label_len.upper),
                audio_transform=post_load_transform,
                label_transform=target_transform,
            ),
            dataset_len=cfg.dataset_len,
        )
    elif supported_dataset == "librispeech":
        cfg = dataset.librispeech
        max_duration = cfg.max_secs.value if cfg.HasField("max_secs") else None
        dataset = LibriSpeech(
            root=cfg.root,
            subsets=[
                cfg.SUBSET.DESCRIPTOR.values_by_number[subset_idx]
                .name.lower()
                .replace("_", "-")
                for subset_idx in cfg.subset
            ],
            pre_load_audio_transforms=pre_load_transform,
            audio_transform=post_load_transform,
            label_transform=target_transform,
            download=download,
            max_duration=max_duration,
        )
    else:
        raise ValueError(f"{supported_dataset} not supported")

    return dataset


def _add_seq_len(transform: Optional[Callable], len_fn: Callable) -> Callable:
    def new_transform(x, *args, **kwargs):
        result = x
        if transform is not None:
            result = transform(result, *args, **kwargs)
        seq_len = torch.tensor(len_fn(result), requires_grad=False)
        return result, seq_len

    return new_transform
