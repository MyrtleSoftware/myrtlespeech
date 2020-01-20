import logging
import os
import tarfile
from typing import List
from typing import Sequence

import torchaudio
from myrtlespeech.data.dataset import utils
from myrtlespeech.data.dataset.base import BaseDataset


class CommonVoice(BaseDataset):
    r"""`CommonVoice Dataset <https://voice.mozilla.org/>`_.

    .. note::

        You must manually download the CommonVoice dataset
        from `https://voice.mozilla.org
        <https://voice.mozilla.org/en/datasets>`_. Mozilla requires that you
        provide an email before the download.

    Args:
        root: Root directory of the dataset. This should contain either a
            compressed .tar file or the ``CommonVoice`` directory with a tsv
            file for each subset that contains paths to samples and
            transcriptions. There should also be a ``clips`` directory
            containing all the audio files. e.g.

            .. code-block::

                CommonVoice/
                    dev.tsv
                    invalidated.tsv
                    other.tsv
                    test.tsv
                    train.tsv
                    validated.tsv
                    clips/
                       common_voice_en_1.mp3
                       common_voice_en_10.mp3
                       common_voice_en_100.mp3
                       ...

        subsets: List of subsets to create the dataset from.  Subset names must
            be one of:

                ``'dev'``
                ``'invalidated'``
                ``'other'``
                ``'test'``
                ``'train'``
                ``'validated'``

        audio_transform: A function that returns a transformed piece of audio
            data.

        label_transform: A function that returns a transformed target.

        download: If :py:data:`True`, dataset is extracted to the ``root``
            directory. Note that automatic download is not supported and the
            user must manually download the .tar file.

        skip_integrity_check: If :py:data:`True` the integrity check is
            skipped.  This is useful when doing quick experiments on the larger
            subsets that can take time to verify and may have been recently
            checked.

        max_duration: All samples with duration (in seconds) greater than this
            will be dropped.

    Attributes:
        VERSION: The CommonVoice version string.
    """

    VERSION: str = "en_1087h_2019-06-12"

    base_dir = "CommonVoice"
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

    def _validate_subsets(self, subsets: Sequence[str]) -> Sequence[str]:
        """Ensures subsets is non-empty and contains valid subsets only."""
        if not subsets:
            raise ValueError("no subsets specified")
        for subset in subsets:
            if subset not in self.subset_hashes.keys():
                raise ValueError(f"{subset} is not valid")
        return subsets

    def download(self) -> None:
        """Extracts dataset unless already cached and performs checksum.

        If the ``clips`` directory and all required dataset files already
        exist, this function becomes a noop.

        If the ``clips`` directory or one of the dataset files are missing then
        the function attempts to find and extract the zipped dataset file.

        If this file is not found, it prompts the user to download the dataset
        from Mozilla's website.
        """
        os.makedirs(os.path.join(self.root, self.base_dir), exist_ok=True)

        already_downloaded = os.path.exists(
            os.path.join(self.root, self.base_dir, "clips")
        )
        for subset in self.subsets:
            already_downloaded &= os.path.isfile(
                os.path.join(self.root, self.base_dir, subset + ".tsv")
            )

        if already_downloaded:
            logging.info("Already downloaded all required data")
            return

        path = os.path.join(self.root, "en.tar.gz")
        if not os.path.isfile(path):
            # Get user to manually download file since Mozilla wants to collect
            # email addresses and agree to not identify speakers when users do
            # this. In the future, we may want to host this ourselves since the
            # dataset is licensed with CC0.
            raise NotImplementedError(
                f"Download dataset version `{CommonVoice.VERSION}` from "
                f"https://voice.mozilla.org/en and place it in {self.root}."
            )

        if utils.checksum_file(path, "md5") != self.archive_hash:
            raise utils.DownloadError(f"invalid checksum for {path}")

        with tarfile.open(path, mode="r|gz") as tar:
            tar.extractall(os.path.join(self.root, self.base_dir))

    def _check_subset_integrity(self, subset: str) -> bool:
        path = os.path.join(self.root, self.base_dir, subset + ".tsv")
        try:
            actual_md5 = utils.checksum_file(path, "md5")
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        return actual_md5 == self.subset_hashes[subset]

    def check_integrity(self) -> None:
        """Returns True if each subset in ``self.subsets`` is valid."""
        super().check_integrity()

        # Checks clips directory
        path = os.path.join(self.root, self.base_dir, "clips")
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

        os.makedirs(
            os.path.join(self.root, self.base_dir, "clips_down"), exist_ok=True
        )
        for subset in self.subsets:
            self._parse_subset_file(
                os.path.join(self.root, self.base_dir), subset
            )

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
        path = os.path.join(root, "clips", file)
        si, _ = torchaudio.info(path)
        duration = (si.length / si.channels) / si.rate
        if self.max_duration is not None and duration > self.max_duration:
            return True

        down_file = file[:-3] + "wav"
        down_path = os.path.join(root, "clips_downsampled", down_file)
        if not os.path.isfile(down_path):
            # Convert mp3 -> wav and downsample to 16KHz
            audio, rate = torchaudio.load(path)
            torchaudio.save(down_path, audio, 16000)
        self.paths.append(down_path)
        self.durations.append(duration)
        return False
