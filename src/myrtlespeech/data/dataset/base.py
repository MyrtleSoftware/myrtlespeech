import os
import warnings
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        subsets: Sequence[str],
        audio_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        label_transform: Optional[Callable[[str], str]] = None,
        download: bool = False,
        skip_integrity_check: bool = False,
        max_duration: Optional[float] = None,
    ):
        self.root = os.path.expanduser(root)
        self.subsets = self._validate_subsets(subsets)
        self._transform = audio_transform
        self._target_transform = label_transform
        self.max_duration = max_duration

        if download:
            self.download()

        if not skip_integrity_check:
            self.check_integrity()
        else:
            warnings.warn("skipping integrity check")

        self.load_data()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        r"""Returns the sample at ``index`` in the dataset.

        The samples are in ascending order by *untransformed* audio sample
        length.

        Untransformed samples are :py:class:`torch.Tensor` with size ``(1,
        samples)``.

        Args:
            index: Index.

        Returns:
            Tuple of ``(audiodata, target)`` where audiodata is the possibly
            transformed audio sample and target is the possibly transformed
            target transcription.
        """
        path = self.paths[index]
        audio, rate = torchaudio.load(path)

        assert rate == 16000, f"{path} sample rate == {rate} != 16000"
        assert (
            audio.size(1) / rate == self.durations[index]
        ), f"{path} sample duration != expected duration"

        if self._transform is not None:
            audio = self._transform(audio)

        target = self.transcriptions[index]
        if self._target_transform is not None:
            target = self._target_transform(target)

        return audio, target

    def __len__(self) -> int:
        return len(self.paths)

    def check_integrity(self) -> None:
        """Returns True if each subset in ``self.subsets`` is valid."""
        for subset in self.subsets:
            if not self._check_subset_integrity(subset):
                raise ValueError(f"subset {subset} not found or corrupt")

    def _process_transcript(self, transcript: str) -> None:
        transcript = transcript.strip().lower()
        self.transcriptions.append(transcript)

    def _sort_by_duration(self) -> None:
        """Orders the loaded data by audio duration, shortest first."""
        total_samples = len(self.paths)
        if total_samples == 0:
            return
        samples = zip(self.paths, self.durations, self.transcriptions)
        sorted_samples = sorted(samples, key=lambda sample: sample[1])
        self.paths, self.durations, self.transcriptions = [
            list(c) for c in zip(*sorted_samples)
        ]
        assert (
            total_samples
            == len(self.paths)
            == len(self.durations)
            == len(self.transcriptions)
        ), "_sort_by_duration len mis-match"

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + "("
            f"root={self.root}, "
            f"subsets={self.subsets})"
        )
