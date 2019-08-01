from typing import List
from typing import Optional


class Alphabet:
    """An alphabet for a language.

    Args:
        symbols: List of symbols in the alphabet. Each symbol will be assigned,
            in iteration order, an index (``int``) starting from 0.

    Raises:
        ``ValueError``: Duplicate symbol in ``symbols``.

    Example:
        >>> Alphabet(symbols=["a", "b", "c", ".", " "])
        Alphabet(symbols=['a', 'b', 'c', '.', ' '])
    """

    def __init__(self, symbols: List[str]):
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbol in symbols.")

        self.symbols = symbols
        self._index_map = dict(enumerate(symbols))
        self._symbol_map = {letter: i for i, letter in self._index_map.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbols={self.symbols})"

    def __len__(self) -> int:
        return len(self.symbols)

    def __getitem__(self, index: int) -> str:
        symbol = self.get_symbol(index)
        if symbol is None:
            raise IndexError("Index %d is out of range")
        return symbol

    def get_symbol(self, index: int) -> Optional[str]:
        """Returns the symbol for an index or None if index has no symbol."""
        return self._index_map.get(index)

    def get_index(self, symbol: str) -> Optional[int]:
        """Returns the index for a symbol or None if symbol not in Alphabet."""
        return self._symbol_map.get(symbol)

    def get_symbols(self, indices: List[int]) -> List[str]:
        """Maps each index in a sequence of indices to it's symbol.

        Args:
            indices: A sequence of indices (int).

        Returns:
            A list of symbols. Indices in the sequence that do not have a
            corresponding symbol will be ignored. This means ``len(returned
            list)`` may be shorter than ``len(indices)``.
        """
        symbols = []
        for index in indices:
            symbol = self.get_symbol(index)
            if symbol is not None:
                symbols.append(symbol)
        return symbols

    def get_indices(self, sentence: List[str]) -> List[int]:
        """Maps each symbol in a sentence to it's index.

        Args:
            sentence: A sequence of symbols.

        Returns:
            A list of indices. Symbols in the sentence that are not in the
            alphabet will be ignored. This means ``len(returned list)`` may be
            shorter than ``len(sentence)``.
        """
        indices = [self.get_index(symbol) for symbol in sentence]
        return list(filter(lambda x: x is not None, indices))  # type: ignore
