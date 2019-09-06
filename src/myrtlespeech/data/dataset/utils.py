import hashlib
import os
from typing import Callable
from typing import Union


INTEGRITY_ALGS = {"md5": hashlib.md5}


class DownloadError(Exception):
    """Exception raised for errors during downloading."""

    pass


def checksum_file(path: Union[bytes, str], algorithm: str) -> str:
    """Returns the checksum of the file at path using the given algorithm.

    Args:
        path: Path of the file to compute the checksum for.
        algorithm: Hash algorithm to use when computing checksum.  See
            ``INTEGRITY_ALGS.keys()`` for a list of supported algorithms.

    Raises:
        :py:class:`ValueError`: ``algorithm`` is not supported.
        :py:class:`FileNotFoundError`: File does not exist at path.
        :py:class:`IsADirectoryError`: ``path`` refers to a directory.
    """
    alg = _parse_algorithm(algorithm)
    accum = alg()
    with open(path, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            accum.update(chunk)
    return accum.hexdigest()


def checksum_dir(root: str, algorithm: str) -> str:
    """Checksums the directory at ``root`` using the given ``algorithm``.

    The checksum is computed in a deterministic way over the directory contents
    at root. Subdirectories are included. Symlinks are not followed.

    Args:
        root: Root of the directory to compute the checksum for.
        algorithm: Hash algorithm to use when computing checksum.  See
            ``INTEGRITY_ALGS.keys()`` for a list of supported algorithms.

    Raises:
        :py:class:`FileNotFoundError`: ``root`` does not exist.
        :py:class:`NotADirectoryError`: ``root`` is not a directory.
        :py:class:`PermissionError`: Inadequate filesystem permissions.
        :py:class:`ValueError`: The stated ``algorithm`` is not supported.
    """
    # Hashing algorithms eat `bytes`. `os.walk` returns `bytes` objects when
    # called with a `bytes` object.
    b_root = root.encode("utf-8")
    alg = _parse_algorithm(algorithm)

    def raise_(err):
        raise err

    accum = alg()
    for cur_root, dirs, files in os.walk(b_root, onerror=raise_):
        dirs.sort()  # Ensures os.walk visits dirs in a deterministic order.
        for file in sorted(files):
            file_path = os.path.join(cur_root, file)
            accum.update(file)
            accum.update(bytes.fromhex(checksum_file(file_path, algorithm)))
        for dir_ in dirs:
            accum.update(dir_)

    return accum.hexdigest()


def _parse_algorithm(algorithm: str) -> Callable:
    """Returns a constructor for the given ``algorithm`` if supported."""
    try:
        alg = INTEGRITY_ALGS[algorithm]
    except KeyError:
        raise ValueError(f"algorithm={algorithm} not supported")
    return alg
