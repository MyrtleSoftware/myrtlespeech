"""Utilities for generating fake data to test out interfaces."""
import math
import random
from dataclasses import dataclass
from typing import Callable
from typing import Generic
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from torch.utils.data import Dataset


SpeechToTextGen = Callable[[int], Tuple[torch.Tensor, str]]


def speech_to_text(
    audio_ms: Tuple[int, int],
    label_symbols: Sequence[str],
    label_len: Tuple[int, int],
    audio_channels: int = 1,
    audio_dtype: torch.dtype = torch.int16,
    audio_device: torch.device = torch.device("cpu"),
    audio_pin_memory: bool = False,
    audio_sample_rate: int = 16000,
    audio_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    label_transform: Optional[Callable[[str], str]] = None,
) -> SpeechToTextGen:
    r"""Returns a function that returns random ``(audio data, string)`` tuples.

    Args:
        audio_ms: Audio sample will have length (time) between this range.
            Range specificed in milliseconds and is inclusive of both ends
            (i.e. a closed interval).

            Requires ``audio_ms[0] <= audio_ms[1]`` and ``audio_ms[0] > 0``.
            i.e. Lower bound must be less than or equal to upper bound and
            non-negative.

            .. note::

                The length of the returned audio array is different: it
                includes ``audio_sample_rate`` samples per second.

        label_symbols: Symbols to sample from when generating the label.

        label_len: The label will have length between this range inclusive.

            Requires ``label_len[0] <= label_len[1]`` and ``label_len[0] >=
            0``.  i.e. Lower bound must be less than or equal to upper bound
            and non-negative.

        audio_channels: Number of audio channels to simulate. Must be ``>= 1``.

        audio_dtype: Data type of the returned audio array. Floating point
            audio data is typically in the range from -1.0 to 1.0. Integer data
            is in the range ``[-2**15, 2**15-1]`` for ``torch.int16`` and from
            ``[-2**31, 2**31-1]`` for ``torch.int32``.

            Must be in ``[torch.float64, torch.float32, torch.int32,
            torch.int16]``.

        audio_device: The device on which each :py:class:`torch.Tensor` of
            audio data will be allocated.

        audio_pin_memory: If :py:const:`True` each :py:class`torch.Tensor` of
            audio data is allocated in the pinned memory. Only works for CPU
            tensors.

        audio_sample_rate: Sample rate specified in Hz.

        audio_transform: If not :py:data:`None` it is a :py:class:`Callable`
            that is applied to the generated audio signal
            (:py:class:`torch.Tensor`) before it is returned.

        label_transform: If not :py:data:`None` it is a :py:class:`Callable`
            that is applied to the generated string before it is returned.

    Returns:
        A function that returns random ``(audio data, string)`` tuples where
        ``audio data`` is a :py:class:`torch.Tensor` with size
        ``(audio_channels, length)`` if ``audio_channels > 1`` else
        ``(length)`` and ``string`` is a ``str``.

        .. Note::

            ``audio_transform`` and ``label_transform`` are applied to ``audio
            data`` and ``string`` respectively provided they are not
            :py:data:`None`.

    Raises:
        :py:class:`ValueError`: if ``audio_ms[0] > audio_ms[1]``.

        :py:class:`ValueError`: if ``audio_ms[0] <= 0``.

        :py:class:`ValueError`: if ``label_len[0] > label_len[1]``.

        :py:class:`ValueError`: if ``label_len[0] < 0``.

        :py:class:`ValueError`: if ``audio_channels < 1``.

        :py:class:`ValueError`: if not in
            ``[torch.float64, torch.float32, torch.int32, torch.int16]``.

    Example:
        >>> generator = speech_to_text(audio_ms=(1, 1000),   # up to 1s long
        ...                            label_symbols="abcd ",
        ...                            label_len=(1, 10))
        >>> audio, label = generator(12345)
        >>> audio.size()   # note: > 1000 due to 16kHz sample rate
        torch.Size([6128])
        >>> label
        'd    bc'

        Note the difference when ``audio_channels > 1``:

        >>> generator = speech_to_text(audio_ms=(1, 1000),   # up to 1s long
        ...                            label_symbols="abcd ",
        ...                            label_len=(1, 10),
        ...                            audio_channels=2)
        >>> audio, label = generator(12345)
        >>> audio.size()   # note: > 1000 due to 16kHz sample rate
        torch.Size([2, 6128])
        >>> label
        'd    bc'
    """
    if audio_ms[0] > audio_ms[1]:
        raise ValueError("audio_ms lower bound must be > upper bound")

    if audio_ms[0] <= 0:
        raise ValueError("audio_ms must be greater than 0")

    if label_len[0] > label_len[1]:
        raise ValueError("label_len lower bound must be > upper bound")

    if label_len[0] < 0:
        raise ValueError("label_len must be greater than or equal to 0")

    if audio_channels < 1:
        raise ValueError("audio_channels must be >= 1")

    dtypes = [torch.float64, torch.float32, torch.int32, torch.int16]
    if audio_dtype not in dtypes:
        raise ValueError(f"audio_dtype must be in {dtypes}")

    rnd = random.Random()

    def generator(key: int) -> Tuple[torch.Tensor, str]:
        rnd.seed(a=key)

        if label_symbols:
            label = "".join(
                rnd.choices(label_symbols, k=rnd.randint(*label_len))
            )
        else:
            label = ""

        audio_samples = math.ceil(
            rnd.randint(*audio_ms) * (audio_sample_rate / 1000)
        )
        if audio_channels > 1:
            audio_size: Union[Tuple[int], Tuple[int, int]] = (
                audio_channels,
                audio_samples,
            )
        else:
            audio_size = (audio_samples,)

        audio = torch.empty(
            audio_size,
            dtype=audio_dtype,
            device=audio_device,
            pin_memory=audio_pin_memory,
        )

        if audio_dtype.is_floating_point:
            audio.normal_(mean=0, std=1)
        elif audio_dtype == torch.int16:
            audio.random_(
                -(2 ** 15), 2 ** 15
            )  # random_ subtracts 1 from upper
        else:
            audio.random_(
                -(2 ** 31), 2 ** 31
            )  # random_ subtracts 1 from upper

        if audio_transform is not None:
            audio = audio_transform(audio)

        if label_transform is not None:
            label = label_transform(label)

        return audio, label

    return generator


FakeT = TypeVar("FakeT")


@dataclass
class FakeDataset(Dataset, Generic[FakeT]):
    r"""A dataset of generated values.

    Samples in a :py:class:`FakeDataset` are generated using ``generator`` each
    time they are accessed rather than being generated and stored when the
    dataset is created. This lazy generation makes it possible to have a large
    ``dataset_len`` whilst keeping initialization times short and memory
    consumption low. Note that because of this accessing the same index twice
    may return two different objects depending on the implementation of
    ``generator``.

    Attributes:
        dataset_len: The size of the dataset.

            >>> dataset = FakeDataset(generator=..., dataset_len=10)
            >>> len(dataset)
            10

        generator: A callable that will return the same value -- but not
            necessarily object -- when called the same ``int`` argument.

            >>> generator = lambda key: key
            >>> dataset = FakeDataset(generator=generator)
            >>> generator(0) == generator(0)
            True
    """
    generator: Callable[[int], FakeT]
    dataset_len: int = 100

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, key: int) -> FakeT:
        if key < 0:
            # enable negative indexing by wrapping such that for i in
            # [1, 2, ..., len(dataset)]:
            #
            #   dataset[len(dataset) - i] == dataset[-i]
            key += self.dataset_len

        if not (0 <= key < self.dataset_len):
            raise IndexError("index out of range")

        # temporarily set seeds to key so the same values are returned for each
        # key without storing the generated tensors
        #
        # rng state saved and then restored whilst modifying seed to avoid
        # causing random events, as much as possible, to become deterministic
        torch_state = torch.get_rng_state()
        torch.manual_seed(key)

        # type disabled due to https://github.com/python/mypy/issues/5485
        value = self.generator(key)  # type: ignore

        torch.set_rng_state(torch_state)

        return value
