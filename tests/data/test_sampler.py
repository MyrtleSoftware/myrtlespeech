from myrtlespeech.data.sampler import SequentialRandomSampler
from myrtlespeech.data.sampler import SortaGrad


def dataset_gen(n_batches, batch_size, full_last_batch):
    """Returns [0, ..., (n_batches*batch_size)-1-int(not(full_last_batch))]"""
    indices = list(range(n_batches * batch_size))
    if not full_last_batch:
        del indices[-1]
    return indices


def test_sorta_grad_correct_len():
    n_batches = 10
    batch_size = 16

    for full_last_batch in [True, False]:
        dataset = sorted(dataset_gen(n_batches, batch_size, full_last_batch))
        sampler = SortaGrad(
            dataset,
            drop_last=full_last_batch,
            batch_size=batch_size,
            shuffle=False,
        )
        assert len(sampler) == n_batches


def test_sorta_grad_batches_non_empty():
    n_batches = 10
    batch_size = 16

    for full_last_batch in [True, False]:
        dataset = sorted(dataset_gen(n_batches, batch_size, full_last_batch))
        sampler = SortaGrad(
            dataset,
            drop_last=full_last_batch,
            batch_size=batch_size,
            shuffle=False,
        )
        for batch in sampler:
            assert len(batch) > 0


def test_sorta_grad_first_pass_sequential_remaining_random():
    n_batches = 10
    batch_size = 16
    dataset = sorted(dataset_gen(n_batches, batch_size, False))

    sortagrad = SortaGrad(
        dataset, drop_last=False, batch_size=batch_size, shuffle=True
    )

    for pass_ in range(100):
        indices = []
        for batch in sortagrad:
            indices.extend(batch)

        assert sorted(indices) == dataset

        if pass_ == 0:
            assert indices == sorted(indices)
        else:
            assert indices != sorted(indices)


def test_sequential_strategy_seq_iter_when_epoch_in_seq_epochs():
    dataset = list(range(10))
    dataset_batches = [[elem] for elem in dataset]
    n_iterators = 5
    sequential = {0, 2, 3, 7, 8, 10}

    seq_strat = SequentialRandomSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        n_iterators=n_iterators,
        sequential=sequential,
    )

    for epoch in range(n_iterators, max(sequential) + 2):
        sampler_batches = [batch for batch in iter(seq_strat)]

        assert len(sampler_batches) == len(dataset_batches)
        assert sorted(sampler_batches) == sorted(dataset_batches)

        if epoch in sequential:
            assert all(
                sample_batch == dataset_batch
                for sample_batch, dataset_batch in zip(
                    sampler_batches, dataset_batches
                )
            )
        else:
            assert not all(
                sample_batch == dataset_batch
                for sample_batch, dataset_batch in zip(
                    sampler_batches, dataset_batches
                )
            )
