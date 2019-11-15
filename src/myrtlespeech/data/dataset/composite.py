from typing import Tuple

import torch
from torch.utils.data import Dataset


class Composite(Dataset):
    """Composite Dataset - combination of multiple individual datasets

    Args:
        *children: the individual datasets to make up this composite one
    """

    def __init__(self, *children: Dataset):
        self.children = children
        self.durations = [
            dur for child in self.children for dur in child.durations
        ]
        self.child_map = [
            (j, i)
            for j, child in enumerate(self.children)
            for i in range(len(child.durations))
        ]
        self._sort_by_duration()

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
        child_id, child_index = self.child_map[index]
        return self.children[child_id][child_index]

    def __len__(self) -> int:
        return len(self.durations)

    def _sort_by_duration(self) -> None:
        """Orders the loaded data by audio duration, shortest first."""
        total_samples = len(self.durations)
        if total_samples == 0:
            return
        samples = zip(self.child_map, self.durations)
        sorted_samples = sorted(samples, key=lambda sample: sample[1])
        self.child_map, self.durations = [
            list(c) for c in zip(*sorted_samples)
        ]
        assert (
            total_samples == len(self.child_map) == len(self.durations)
        ), "_sort_by_duration len mis-match"

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "["
            + ",".join([child.__repr__() for child in self.children])
            + "]"
        )
