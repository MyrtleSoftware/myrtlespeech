from typing import Tuple, List, Union

import hypothesis.strategies as st
import pytest
from hypothesis import given
from mypy_extensions import TypedDict

from myrtlespeech.data.alphabet import Alphabet


# Fixtures and Strategies -----------------------------------------------------

AlphabetKwargs = TypedDict("AlphabetKwargs", {"symbols": List[str]})


@st.composite
def random_alphabet(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[Alphabet],
    st.SearchStrategy[Tuple[Alphabet, AlphabetKwargs]],
]:
    """Returns a SearchStrategy for an Alphabet plus maybe the kwargs."""
    kwargs = {"symbols": list(draw(st.sets(elements=st.characters())))}
    alphabet = Alphabet(**kwargs)
    if return_kwargs:
        return alphabet, kwargs
    return alphabet


@st.composite
def duplicate_symbols(draw) -> st.SearchStrategy[List[str]]:
    """Returns a SearchStrategy for a list of chars where >=1 is duplicated."""
    symbols = draw(st.lists(st.characters(), min_size=1))

    dup_idx = draw(st.integers(min_value=0, max_value=len(symbols) - 1))
    ins_idx = draw(st.integers(min_value=0, max_value=len(symbols)))

    symbols = symbols[:ins_idx] + [symbols[dup_idx]] + symbols[ins_idx:]

    return symbols


# Tests -----------------------------------------------------------------------


@given(alphabet=random_alphabet())
def test_repr_does_not_raise_error(alphabet: Alphabet) -> None:
    """Ensures ``__repr__`` does not raise an error."""
    repr(alphabet)


@given(symbols=duplicate_symbols())
def test_duplicate_symbol_raise_value_error(symbols: List[str]) -> None:
    """Ensures ``ValueError`` raised when ``Alphabet`` init has dup. symbols."""
    with pytest.raises(ValueError):
        Alphabet(symbols)


@given(alphabet_kwargs=random_alphabet(return_kwargs=True))
def test_len(alphabet_kwargs: Tuple[Alphabet, AlphabetKwargs]) -> None:
    """Ensures ``len(alphabet)`` is correct."""
    assert len(alphabet_kwargs[0]) == len(alphabet_kwargs[1]["symbols"])


@given(alphabet_kwargs=random_alphabet(return_kwargs=True))
def test_iterator(alphabet_kwargs: Tuple[Alphabet, AlphabetKwargs]) -> None:
    """Ensures iter over ``Alphabet`` is same as iter over ``symbols``."""
    alphabet, kwargs = alphabet_kwargs
    for index, symbol in enumerate(alphabet):
        assert symbol == kwargs["symbols"][index]


@given(alphabet_kwargs=random_alphabet(return_kwargs=True))
def test_get_symbol(alphabet_kwargs: Tuple[Alphabet, AlphabetKwargs]) -> None:
    """Ensures ``Alphabet.get_symbol`` returns the correct symbol."""
    alphabet, kwargs = alphabet_kwargs
    for index, symbol in enumerate(kwargs["symbols"]):
        assert alphabet.get_symbol(index) == symbol


@given(alphabet_kwargs=random_alphabet(return_kwargs=True))
def test_get_index(alphabet_kwargs: Tuple[Alphabet, AlphabetKwargs]) -> None:
    """Ensures ``Alphabet.get_index`` returns the correct index."""
    alphabet, kwargs = alphabet_kwargs
    for index, symbol in enumerate(kwargs["symbols"]):
        assert alphabet.get_index(symbol) == index


@given(data=st.data(), alphabet_kwargs=random_alphabet(return_kwargs=True))
def test_get_symbols(data, alphabet_kwargs: Tuple[Alphabet, AlphabetKwargs]):
    """Ensures ``Alphabet.get_symbols`` returns all valid symbols."""
    alphabet, kwargs = alphabet_kwargs

    indices = data.draw(st.lists(elements=st.integers()))

    sentence = []
    for idx in indices:
        symbol = alphabet.get_symbol(idx)
        if symbol is not None:
            sentence.append(symbol)

    actual = alphabet.get_symbols(indices)

    assert len(actual) == len(sentence)
    assert all([a == s for a, s in zip(actual, sentence)])


@given(data=st.data(), alphabet_kwargs=random_alphabet(return_kwargs=True))
def test_get_indices(data, alphabet_kwargs: Tuple[Alphabet, AlphabetKwargs]):
    """Ensures ``Alphabet.get_indices`` returns all valid indices."""
    alphabet, kwargs = alphabet_kwargs

    one_of_elements = [st.characters()]
    if len(kwargs["symbols"]) > 0:
        one_of_elements.append(st.sampled_from(kwargs["symbols"]))

    sentence = data.draw(st.lists(elements=st.one_of(one_of_elements)))

    indices = []
    for symbol in sentence:
        idx = alphabet.get_index(symbol)
        if idx is not None:
            indices.append(idx)

    actual = alphabet.get_indices(sentence)

    assert len(actual) == len(indices)
    assert all([a == i for a, i in zip(actual, indices)])
