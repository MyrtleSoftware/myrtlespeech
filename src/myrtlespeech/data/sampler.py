import random


class RandomBatchSampler:
    """TODO"""

    def __init__(self, indices, batch_size, shuffle, drop_last=False):
        self.shuffle = shuffle
        self.batch_indices = self._batch_indices(
            indices, batch_size, drop_last
        )

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
        if self.shuffle:
            random.shuffle(self.batch_indices)
        for b in self.batch_indices:
            yield b

    def __len__(self):
        return len(self.batch_indices)
