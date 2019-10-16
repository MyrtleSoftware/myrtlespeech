import fnmatch
import os
import shutil
import tarfile
import warnings
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import requests
import torch
import torchaudio
from myrtlespeech.data.dataset import utils
from torch.utils.data import Dataset


class CommonVoice(Dataset):
    """CommonVoice Dataset - https://voice.mozilla.org/en

    Args:
        root: Root directory of the dataset. This should contain the
            ``CommonVoice`` directory which itself contains a clips directory
            containing all voice samples and a tsv file for each subset with
            paths to samples and their transcription

        subsets: List of subsets to create the dataset from.  Subset names must
            be one of:

                ``'other'``
                ``'train'``
                ``'dev'``
                ``'invalidated'``
                ``'test'``
                ``'validated``

        audio_transform: A function that returns a transformed piece of audio
            data.

        label_transform: A function that returns a transformed target.

        download: If :py:data:`True`, downloads the dataset from the internet
            and extracts it in the ``root`` directory. If the dataset is
            already downloaded, it is not downloaded again. See the
            :py:meth:`CommonVoice.download` method for more information.

        skip_integrity_check: If :py:data:`True` the integrity check is skipped.
            This is useful when doing quick experiments on the larger subsets
            that can take time to verify and may have been recently checked.

        max_duration: All samples with duration (in seconds) greater than this
            will be dropped.
    """

    base_dir = "clips"
    subset_hashes = {
        "other": "ede180d95179e234e6dd0a9c30bcc88f",
        "train": "d5c627cad5d0f9523e5f0a362123a092",
        "dev": "19a0f8a071b4402207052d0caa73beea",
        "invalidated": "da059710646c40e5a5126429ec99c6d6",
        "test": "46766f72678b1c20576e528fc0c8dac2",
        "validated": "67f103ff2bd4a3f774bd67962454bbda",
    }
    clips_hash = "960f1ac38efa1b8121439a09265f7619"
    archive_hash = "602e73387431ba90afb2912cf59eb03f"

    def __init__(
        self,
        root: str,
        subsets: Sequence[str],
        audio_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        label_transform: Optional[Callable[[str], str]] = None,
        download: bool = False,
        skip_integrity_check: bool = False,
        max_duration: Optional[float] = None,
    ):
        self.root = os.path.expanduser(root)
        self.subsets = self._validate_subsets(subsets)
        self._transform = audio_transform
        self._target_transform = label_transform
        self.max_duration = max_duration

        if download:
            self.download()

        if not skip_integrity_check:
            self.check_integrity()
        else:
            warnings.warn("skipping integrity check")

        self.load_data()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        r"""Returns the sample at ``index`` in the dataset.

        The samples are in ascending order by *untransformed* audio sample
        length.

        Untransformed samples are :py:class:`torch.Tensor` with size ``(1,
        samples)``.

        Args:
            index: Index.

        Returns:
            Tuple of ``(audiodata, target)`` where audiodata is the possibly
            transformed audio sample and target is the possibly transformed
            target transcription.
        """
        path = self.paths[index]
        audio, rate = torchaudio.load(path)
        print(audio.size(1))

        assert rate == 48000, f"{path} sample rate == {rate} != 48000"
        # Add offset for 48KHz files since torchaudio is inconsistent with lengths:
        # https://github.com/pytorch/audio/issues/329
        assert (audio.size(1) + 1152) / rate == self.durations[
            index
        ], f"{path} sample duration != expected duration"

        if self._transform is not None:
            audio = self._transform(audio)

        target = self.transcriptions[index]
        if self._target_transform is not None:
            target = self._target_transform(target)

        return audio, target

    def __len__(self) -> int:
        return len(self.paths)

    def _validate_subsets(self, subsets: Sequence[str]) -> Sequence[str]:
        """Ensures subsets is non-empty and contains valid subsets only."""
        if not subsets:
            raise ValueError("no subsets specified")
        for subset in subsets:
            if subset not in self.subset_hashes.keys():
                raise ValueError(f"{subset} is not valid")
        return subsets

    def download(self) -> None:
        """Extracts data set unless already cached.

        If the clips directory and all required data set files already exist, this
        function becomes a noop

        If the clips directory or one of the data set files are missing then the function
        attempts to find and extract the zipped data set file

        If this file is not found, it prompts the user to download the data set from
        Mozilla's website
        """
        os.makedirs(self.root, exist_ok=True)

        alread_downloaded = os.path.exists(
            os.path.join(self.root, self.base_dir)
        )
        for subset in self.subsets:
            alread_downloaded &= os.path.isfile(
                os.path.join(self.root, subset + ".tsv")
            )

        if alread_downloaded:
            print("Already downloaded all required data")
            return

        path = os.path.join(self.root, "en.tar.gz")
        if not os.path.isfile(path):
            # Get user to manually download file since Mozilla wants to collect email addresses
            # and agree to not identify speakers when users do this
            # In the future, we may want to host this ourselves since the data set is licensed
            # with CC0
            raise NotImplementedError(
                f"Please download the data set from https://voice.mozilla.org/en and"
                f"place it in {self.root}"
            )

        if utils.checksum_file(path, "md5") != self.archive_hash:
            raise utils.DownloadError(f"invalid checksum for {path}")

        with tarfile.open(path, mode="r|gz") as tar:
            tar.extractall(self.root)

    def _check_subset_integrity(self, subset: str) -> bool:
        path = os.path.join(self.root, subset + ".tsv")
        try:
            actual_md5 = utils.checksum_file(path, "md5")
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        return actual_md5 == self.subset_hashes[subset]

    def check_integrity(self) -> None:
        """Returns True if each subset in ``self.subsets`` is valid."""
        for subset in self.subsets:
            if not self._check_subset_integrity(subset):
                raise ValueError(f"subset {subset} not found or corrupt")

        # Checks clips directory
        path = os.path.join(self.root, self.base_dir)
        try:
            actual_md5 = utils.checksum_dir(path, "md5")
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            raise ValueError(f"{self.base_dir} directory not found")
        if actual_md5 != self.clips_hash:
            raise ValueError(f"{self.base_dir} directory corrupt")

    def load_data(self) -> None:
        """Loads the data from disk."""
        self.paths: List[str] = []
        self.durations: List[float] = []
        self.transcriptions: List[str] = []

        for subset in self.subsets:
            self._parse_subset_file(self.root, subset)

        self._sort_by_duration()

    def _parse_subset_file(self, root: str, name: str) -> None:
        """Parses each sample in a transcription file."""
        trans_path = os.path.join(root, name + ".tsv")
        with open(trans_path, "r", encoding="utf-8") as trans:
            trans.readline()  # First line is header
            # Each line has the form "ID \t PATH \t TRANSCRIPT \t <METADATA>"
            for line in trans:
                parts = line.split("\t")
                path = parts[1]
                transcript = parts[2]
                dropped = self._process_audio(root, path)
                if not dropped:
                    self._process_transcript(transcript)

    def _process_audio(self, root: str, file: str) -> bool:
        """Returns True if sample was dropped due to being too long."""
        path = os.path.join(root, self.base_dir, file)
        si, _ = torchaudio.info(path)
        duration = (si.length / si.channels) / si.rate
        if self.max_duration is not None and duration > self.max_duration:
            return True
        self.paths.append(path)
        self.durations.append(duration)
        return False

    def _process_transcript(self, transcript: str) -> None:
        transcript = transcript.strip().lower()
        self.transcriptions.append(transcript)

    def _sort_by_duration(self) -> None:
        """Orders the loaded data by audio duration, shortest first."""
        total_samples = len(self.paths)
        if total_samples == 0:
            return
        samples = zip(self.paths, self.durations, self.transcriptions)
        sorted_samples = sorted(samples, key=lambda sample: sample[1])
        self.paths, self.durations, self.transcriptions = [
            list(c) for c in zip(*sorted_samples)
        ]
        assert (
            total_samples
            == len(self.paths)
            == len(self.durations)
            == len(self.transcriptions)
        ), "_sort_by_duration len mis-match"

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + "("
            f"root={self.root}, "
            f"subsets={self.subsets})"
        )
