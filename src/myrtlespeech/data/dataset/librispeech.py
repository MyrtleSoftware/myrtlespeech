import fnmatch
import logging
import os
import shutil
import tarfile
from typing import List
from typing import Sequence

import requests
import torchaudio
from myrtlespeech.data.dataset import utils
from myrtlespeech.data.dataset.base import BaseDataset


class LibriSpeech(BaseDataset):
    """LibriSpeech Dataset - http://openslr.org/12/

    Args:
        root: Root directory of the dataset. This should contain the
            ``LibriSpeech`` directory which itself contains one directory per
            subset in subsets.  These will be created if they do not exist, or
            are corrupt, and download is True.

        subsets: List of subsets to create the dataset from.  Subset names must
            be one of:

                ``'train-clean-100'``
                ``'train-clean-360'``
                ``'train-other-500'``
                ``'dev-clean'``
                ``'dev-other'``
                ``'test-clean``
                ``'test-other'``

        audio_transform: A function that returns a transformed piece of audio
            data.

        label_transform: A function that returns a transformed target.

        download: If :py:data:`True`, downloads the dataset from the internet
            and extracts it in the ``root`` directory. If the dataset is
            already downloaded, it is not downloaded again. See the
            :py:meth:`LibriSpeech.download` method for more information.

        skip_integrity_check: If :py:data:`True` the integrity check is
            skipped. This is useful when doing quick experiments on the larger
            subsets that can take time to verify and may have been recently
            checked.

        max_duration: All samples with duration (in seconds) greater than this
            will be dropped.
    """

    base_dir = "LibriSpeech"
    openslr_url = "http://www.openslr.org/resources/12/"
    data_files = {
        "train-clean-100": {
            "archive_md5": "2a93770f6d5c6c964bc36631d331a522",
            "dir_md5": "b1b762a7384c17c06eee975933c71739",
        },
        "train-clean-360": {
            "archive_md5": "c0e676e450a7ff2f54aeade5171606fa",
            "dir_md5": "ef1c8b93522d89ae27116d64a12d1d2f",
        },
        "train-other-500": {
            "archive_md5": "d1a0fd59409feb2c614ce4d30c387708",
            "dir_md5": "851de4dff9bdfd1a89d9eab18bc675c6",
        },
        "dev-clean": {
            "archive_md5": "42e2234ba48799c1f50f24a7926300a1",
            "dir_md5": "9e3b56b96e2cbbcc941c00f52f2fdcf9",
        },
        "dev-other": {
            "archive_md5": "c8d0bcc9cca99d4f8b62fcc847357931",
            "dir_md5": "1417e91c9f0a1c2c1c61af1178ffa94b",
        },
        "test-clean": {
            "archive_md5": "32fa31d27d2e1cad72775fee3f4849a9",
            "dir_md5": "5fad2e72ec7af2659d50e4df720bc22b",
        },
        "test-other": {
            "archive_md5": "fb5a50374b501bb3bac4815ee91d3135",
            "dir_md5": "ddcbdd339bd02c8d2b6c1bbde28c828c",
        },
    }

    def _validate_subsets(self, subsets: Sequence[str]) -> Sequence[str]:
        """Ensures subsets is non-empty and contains valid subsets only."""
        if not subsets:
            raise ValueError("no subsets specified")
        for subset in subsets:
            if subset not in self.data_files.keys():
                raise ValueError(f"{subset} is not valid")
        return subsets

    def download(self) -> None:
        """Downloads and extracts ``self.subsets`` unless already cached.

        For each subset there are 3 possibilities:

            1. The ``LibriSpeech/subset`` directory exists and is valid making
               this function a noop.

            2. If not 1. but the ``subset.tar.gz`` archive file exists and is
               valid then its contents are extracted.

            3. If not 2. then ``subset.tar.gz`` is downloaded, checksum
               verified, and its contents extracted. The archive file is then
               removed leaving behind the ``LibriSpeech/subset`` directory.
        """
        os.makedirs(self.root, exist_ok=True)

        for subset in self.subsets:
            if self._check_subset_integrity(subset):
                logging.info(f"{subset} already downloaded and verified")
                continue
            path = os.path.join(self.root, subset + ".tar.gz")

            already_present = os.path.isfile(path)
            if not already_present:
                subset_url = self.openslr_url + subset + ".tar.gz"
                with requests.get(subset_url, stream=True) as r:
                    r.raise_for_status()
                    with open(path, "wb") as f:
                        shutil.copyfileobj(r.raw, f)

            archive_md5 = self.data_files[subset]["archive_md5"]
            if utils.checksum_file(path, "md5") != archive_md5:
                raise utils.DownloadError(f"invalid checksum for {path}")

            with tarfile.open(path, mode="r|gz") as tar:
                tar.extractall(self.root)

            if not already_present:
                os.remove(path)

    def _check_subset_integrity(self, subset: str) -> bool:
        path = os.path.join(self.root, self.base_dir, subset)
        try:
            actual_md5 = utils.checksum_dir(path, "md5")
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        return actual_md5 == self.data_files[subset]["dir_md5"]

    def load_data(self) -> None:
        """Loads the data from disk."""
        self.paths: List[str] = []
        self.durations: List[float] = []
        self.transcriptions: List[str] = []

        def raise_(err):
            """raises error if problem during os.walk"""
            raise err

        for subset in self.subsets:
            subset_path = os.path.join(self.root, self.base_dir, subset)
            for root, dirs, files in os.walk(subset_path, onerror=raise_):
                if not files:
                    continue
                matches = fnmatch.filter(files, "*.trans.txt")
                assert len(matches) == 1, "> 1 transcription file found"
                self._parse_transcription_file(root, matches[0])

        self._sort_by_duration()

    def _parse_transcription_file(self, root: str, name: str) -> None:
        """Parses each sample in a transcription file."""
        trans_path = os.path.join(root, name)
        with open(trans_path, "r", encoding="utf-8") as trans:
            # Each line has the form "ID THE TARGET TRANSCRIPTION"
            for line in trans:
                id_, transcript = line.split(maxsplit=1)
                dropped = self._process_audio(root, id_)
                if not dropped:
                    self._process_transcript(transcript)

    def _process_audio(self, root: str, id: str) -> bool:
        """Returns True if sample was dropped due to being too long."""
        path = os.path.join(root, id + ".flac")
        si, _ = torchaudio.info(path)
        duration = (si.length / si.channels) / si.rate
        if self.max_duration is not None and duration > self.max_duration:
            return True
        self.paths.append(path)
        self.durations.append(duration)
        return False
