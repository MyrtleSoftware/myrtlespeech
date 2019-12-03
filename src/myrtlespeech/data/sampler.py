import random


class SequentialRandomSampler:
    """A sequential or random iterable over batches.

    The iterator used each time this iterable is iterated over will yield
    batches either sequentially (i.e. in-order) or randomly (uniform without
    replacement) from `batches`.
    This iterable records the number of times it has returned an iterator. A
    sequential iterator is returned if the current count is in `sequential`.

    Args:
        indices: data with which batches are created.
        batch_size: Batch dimension.
        shuffle: Set to True to have the data reshuffled at every epoch if a
            random iterator is used.
        drop_last: (optional, bool): Set to True to drop the last incomplete
            batch, if the dataset size is not divisible by the batch size. If
            False and the size of dataset is not divisible by the batch size,
            then the last batch will be smaller.
        n_iterators (optional, int): Number of iterators returned so far.
        sequential (optional, set of int): Counts at which to return a
            sequential iterator.

    Yields:
        Batches from `batches`.
    """

    def __init__(
        self,
        indices,
        batch_size,
        shuffle,
        drop_last=False,
        n_iterators=0,
        sequential=None,
    ):
        self.shuffle = shuffle
        self.batch_indices = self._batch_indices(
            indices, batch_size, drop_last
        )
        self._n_iterators = n_iterators
        self._sequential = sequential or {}

    def _batch_indices(self, indices, batch_size, drop_last):
        batches = []
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        if batch and not drop_last:
            batches.append(batch)
        return batches

    def __iter__(self):
        indices = list(range(len(self.batch_indices)))
        if self.shuffle and self._n_iterators not in self._sequential:
            random.shuffle(indices)
        self._n_iterators += 1
        for index in indices:
            yield self.batch_indices[index]

    def __len__(self):
        return len(self.batch_indices)


class SortaGrad(SequentialRandomSampler):
    """An iterable over batch indices according to the SortaGrad strategy.

    The SortaGrad curriculum learning strategy iterates over batches from the
    batched dataset sequentially for the first pass and then randomly for all
    other passes. See Deep Speech 2 paper for more information on this:
    https://arxiv.org/abs/1512.02595

    Args:
        indices: data with which batches are created.
        batch_size: Batch dimension.
        shuffle: Set to True to have the data reshuffled at every epoch if a
            random iterator is used.
        drop_last: (optional, bool): Set to True to drop the last incomplete
            batch, if the dataset size is not divisible by the batch size. If
            False and the size of dataset is not divisible by the batch size,
            then the last batch will be smaller.
        start_epoch (optional, int): Number of iterators returned so far by the
            sampler.

    Yields:
        Batches from `batches`.
    """

    def __init__(
        self, indices, batch_size, shuffle, drop_last=False, start_epoch=0
    ):
        super().__init__(
            indices,
            batch_size,
            shuffle,
            drop_last,
            n_iterators=start_epoch,
            sequential={0},
        )
