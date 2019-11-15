from collections.abc import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Calculates the Levenshtein distance between a and b.

        *This is a straightforward implementation of a well-known algorithm,
        and thus probably shouldn't be covered by copyright to begin with. But
        in case it is, the author (Magnus Lie Hetland) has, to the extent
        possible under law, dedicated all copyright and related and neighboring
        rights to this software to the public domain worldwide, by distributing
        it under the CC0 license, version 1.0. This software is distributed
        without any warranty.  For more information, see*
        <http://creativecommons.org/publicdomain/zero/1.0>

        -- `Magnug Lie Hetland <https://folk.idi.ntnu.no/mlh/hetland_org/\
coding/python/levenshtein.py>`_

    Args:
        a: A :py:class:`Sequence` that supports equality (e.g.
            :py:meth:`object.__eq__`).

        b: A :py:class:`Sequence` that supports equality (e.g.
            :py:meth:`object.__eq__`).

    Returns:
        An integer giving the minimum number of edits (insertions, deletions or
        substitutions) required to change one sequence to the other.

    Example:
        >>> a = "hello world"
        >>> b = "hella world"
        >>> levenshtein(a, b)
        1
        >>> a = ["hello", "world"]
        >>> b = ["hallo", "world", "!"]
        >>> levenshtein(a, b)
        2
        >>> a = [1, 2, 3, 4, 5]
        >>> b = []
        >>> levenshtein(a, b)
        5
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
