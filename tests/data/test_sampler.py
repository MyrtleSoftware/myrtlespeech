from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
from hypothesis import given
from myrtlespeech.data.sampler import SequentialRandomSampler
from myrtlespeech.data.sampler import SortaGrad


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def dataset_gen(
    draw, return_kwargs: bool = False
) -> Union[st.SearchStrategy[List], st.SearchStrategy[Tuple[List, Dict]]]:
    """Returns a SearchStrategy for a list of dataset indices."""
    kwargs = {}
    kwargs["n_batches"] = draw(st.integers(10, 20))
    kwargs["batch_size"] = draw(st.integers(2, 16))
    kwargs["full_last_batch"] = draw(st.booleans())

    # indices = [0, ..., (n_batches*batch_size)-1-int(not(full_last_batch))]
    indices = list(range(kwargs["n_batches"] * kwargs["batch_size"]))
    if not kwargs["full_last_batch"]:
        del indices[-1]

    if not return_kwargs:
        return indices
    return indices, kwargs


@st.composite
def sequential_epochs_gen(draw) -> st.SearchStrategy[List]:
    """Returns a SearchStrategy for a list of sequential epoch numbers."""
    max_size = draw(st.integers(min_value=1, max_value=10))

    sequential = draw(
        st.lists(
            elements=st.integers(min_value=11, max_value=20),
            min_size=1,
            max_size=max_size,
            unique=True,
        )
    )

    return sequential


# Tests -----------------------------------------------------------------------


@given(dataset_kwargs=dataset_gen(return_kwargs=True))
def test_sorta_grad_correct_len(dataset_kwargs: Tuple[List, Dict]):
    dataset, kwargs = dataset_kwargs

    dataset = sorted(dataset)
    sampler = SortaGrad(
        dataset,
        drop_last=kwargs["full_last_batch"],
        batch_size=kwargs["batch_size"],
        shuffle=False,
    )
    assert len(sampler) == kwargs["n_batches"]


@given(dataset_kwargs=dataset_gen(return_kwargs=True))
def test_sorta_grad_batches_non_empty(dataset_kwargs: Tuple[List, Dict]):
    dataset, kwargs = dataset_kwargs

    dataset = sorted(dataset)
    sampler = SortaGrad(
        dataset,
        drop_last=kwargs["full_last_batch"],
        batch_size=kwargs["batch_size"],
        shuffle=False,
    )
    for batch in sampler:
        assert len(batch) > 0


@given(dataset_kwargs=dataset_gen(return_kwargs=True))
def test_sorta_grad_first_pass_sequential_remaining_random(
    dataset_kwargs: Tuple[List, Dict]
):
    dataset, kwargs = dataset_kwargs

    dataset = sorted(dataset)
    sortagrad = SortaGrad(
        dataset,
        drop_last=kwargs["full_last_batch"],
        batch_size=kwargs["batch_size"],
        shuffle=True,
    )

    indices: list
    for pass_ in range(100):
        indices = []
        for batch in sortagrad:
            indices.extend(batch)

        assert sorted(indices) == dataset

        if pass_ == 0:
            assert indices == sorted(indices)
        else:
            assert indices != sorted(indices)


@given(
    dataset_kwargs=dataset_gen(return_kwargs=True),
    n_iterators=st.integers(min_value=1, max_value=10),
    sequential=sequential_epochs_gen(),
)
def test_sequential_strategy_seq_iter_when_epoch_in_seq_epochs(
    dataset_kwargs: Tuple[List, Dict], n_iterators: int, sequential: List
):
    dataset, kwargs = dataset_kwargs

    dataset_batches = []
    batch = []
    for elem in dataset:
        batch.append(elem)
        if len(batch) == kwargs["batch_size"]:
            dataset_batches.append(batch)
            batch = []
    if batch and not kwargs["full_last_batch"]:
        dataset_batches.append(batch)
    sequential_epochs = set(sorted(sequential))

    seq_strat = SequentialRandomSampler(
        dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        n_iterators=n_iterators,
        sequential=sequential_epochs,
    )

    for epoch in range(n_iterators, max(sequential_epochs) + 2):
        sampler_batches = [batch for batch in iter(seq_strat)]

        assert len(sampler_batches) == len(dataset_batches)
        assert sorted(sampler_batches) == sorted(dataset_batches)

        if epoch in sequential_epochs:
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
