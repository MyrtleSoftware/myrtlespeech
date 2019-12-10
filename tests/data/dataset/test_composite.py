from typing import Sequence
from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from myrtlespeech.data.dataset.composite import Composite
from torch.utils.data import Dataset


# Fixtures and Strategies -----------------------------------------------------


class MockDataset(Dataset):
    def __init__(self, durations: Sequence[float]):
        self.durations = sorted(durations)
        self.called = [0] * len(durations)

    def __len__(self) -> int:
        return len(self.durations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        self.called[idx] += 1
        return torch.tensor(self.durations[idx]), str(idx)


@st.composite
def mock_dataset_strategy(draw) -> st.SearchStrategy[MockDataset]:
    """Defines a SearchStrategy for MockDatasets."""
    durations = draw(st.lists(st.floats(1, 20), 1, 10))
    return MockDataset(durations)


@st.composite
def composite_dataset_strategy(
    draw,
) -> st.SearchStrategy[Tuple[Composite, Sequence[MockDataset]]]:
    """Defines a SearchStrategy for CompositeDatasets made of MockDatasets."""
    num_children = draw(st.integers(1, 4))
    children = [draw(mock_dataset_strategy()) for _ in range(num_children)]
    return Composite(*children), children


# Tests -----------------------------------------------------------------------


@given(composite_dataset_strategy())
def test_composite_length_sum_of_children(data):
    comp, children = data
    assert len(comp) == sum(
        len(child) for child in children
    ), "Composite contains a different number of elements to its children"


@given(composite_dataset_strategy())
def test_composite_durations_sorted_order(data):
    comp, _ = data
    dur = comp.durations
    assert all(
        dur[i] <= dur[i + 1] for i in range(len(dur) - 1)
    ), "durations in Composite not in sorted order"


@given(composite_dataset_strategy())
def test_composite_visits_every_elm_once(data):
    comp, children = data
    for _ in comp:
        pass

    assert all(
        all(call == 1 for call in child.called) for child in children
    ), "some child elements not visited once when iterating through Composite"


@given(composite_dataset_strategy())
def test_composite_visits_elems_in_order(data):
    comp, _ = data
    durations = comp.durations
    visited = []
    for x, _ in comp:
        visited.append(x.item())

    assert (
        # Allow minor differences due to rounding
        np.allclose(durations, visited)
    ), "elements of Composite not visited in increasing duration order"
