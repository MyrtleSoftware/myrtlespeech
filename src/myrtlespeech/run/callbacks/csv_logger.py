import csv
import pathlib
from collections.abc import Container
from datetime import datetime
from typing import Optional
from typing import Union

from myrtlespeech.run.callbacks.callback import Callback


class CSVLogger(Callback):
    r"""Logs at the end of an epoch in CSV format.

    This callback saves the CSV to ``log_dir`` with file name
    ``log_<datetime.isoformat>.csv`` when
    :py:meth:`.CallbackHandler.on_train_begin` is called.

    The first entry in each row of the CSV file denotes the date and time the
    entry was logged in ISO 8601 format (:py:meth:`datetime.isoformat`). The
    second entry in each row of the CSV file will be a string denoting the
    current stage. This will either be ``train`` or ``eval``.

    Args:
        log_dir: A pathlike object representing the directory to which results
            will be logged.

        keys: Keys in :py:data:`.CallbackHandler.state_dict`, passed as
            ``**kwargs`` to the ``CSVLogger``, whose values will be logged. If
            :py:data:`None` then all keys in
            :py:data:`.CallbackHandler.state_dict` when
            :py:meth:`.CallbackHandler.on_train_begin` is called are logged.

            A key can refer to nested dictionaries through the use of a ``/``.
            For example the key ``"foo/bar/baz/bang"`` refers to
            ``kwargs["foo"]["bar"]["baz"]["bang"]``.

        exclude: Keys to exclude from the logs. The first and second entires --
            date + time and stage -- cannot be excluded.

    Example:
        >>> # imports
        >>> import tempfile
        >>> import glob
        >>> from myrtlespeech.run.callbacks.callback import CallbackHandler
        >>>
        >>> # create file to write to
        >>> temp = tempfile.TemporaryDirectory()
        >>>
        >>> # initialize CSVLogger and CallbackHandler
        >>> csv_logger = CSVLogger(log_dir=temp.name, keys=["epoch"])
        >>> cb_handler = CallbackHandler(callbacks=[csv_logger])
        >>>
        >>> # simulate training and eval example
        >>> cb_handler.on_train_begin(epochs=1)
        >>> _ = cb_handler.train(mode=True)
        >>> _ = cb_handler.on_epoch_end()
        >>> _ = cb_handler.train(mode=False)
        >>> _ = cb_handler.on_epoch_end()
        >>>
        >>> # read and print file
        >>> # note the first entry in each CSV row is the date
        >>> # this will change per each doctest invocation so it is ignored
        >>> # using ellipsis (...)
        >>> csv_name = glob.glob(f'{temp.name}/*.csv')[0]
        >>> csv_contents = open(csv_name, 'r').read()
        >>> print("example_output:\n", csv_contents)  # doctest:+ELLIPSIS
        example_output:
        ...,stage,epoch
        ...,train,0
        ...,eval,1
        <BLANKLINE>
    """

    def __init__(
        self,
        log_dir: Union[str, pathlib.Path],
        keys: Optional[Container] = None,
        exclude: Optional[Container] = None,
    ):
        super().__init__()
        self.path = (
            pathlib.Path(log_dir) / f"log_{datetime.now().isoformat()}.csv"
        )
        self.keys = keys
        self.exclude = set() if exclude is None else exclude

    def _get_reports(self, reports, path=""):
        """Yields (key, val) tuples from reports, excluding keys in exclude."""
        # invariant: path is either empty ("") or ends in a "/"
        for metric, value in reports.items():
            if path + metric in self.exclude:
                continue
            if isinstance(value, dict):
                yield from self._get_reports(value, path=path + metric + "/")
            else:
                yield path + metric, repr(value)

    def on_train_begin(self, **kwargs):
        """Initializes ``CSVLogger.keys`` if ``None`` and writes CSV header."""
        if self.keys is None:
            self.keys = [metric for metric, _ in self._get_reports(kwargs)]
        keys = ["datetime.isoformat", "stage"] + self.keys
        with self.path.open("w", newline="") as f:
            csv.writer(f).writerow(keys)

    def on_epoch_end(self, **kwargs):
        """Writes current stage and ``CSVLogger.keys`` values to CSV file."""
        vals = dict(self._get_reports(kwargs))
        if "stage" not in vals:
            vals["stage"] = "train" if self.training else "eval"
        if "datetime.isoformat" not in vals:
            vals["datetime.isoformat"] = datetime.now().isoformat()
        keys = ["datetime.isoformat", "stage"] + self.keys
        vals = [vals[k] for k in keys]
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(vals)
