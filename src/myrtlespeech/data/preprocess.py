"""
Utilities for preprocessing audio data.
"""
from typing import Tuple

import torch
import python_speech_features


class MFCC:
    """Compute the Mel-frequency cepstral coefficients (MFCC) of audiodata.

    Args:
        numcep: Number of cepstrum to return.

        winlen: Length of the analysis window in seconds.

        winstep: Step between successive windows in seconds.

        sample_rate: Sample rate of the audio signal.

    Example:
        >>> MFCC(numcep=26, winlen=0.02, winstep=0.01, sample_rate=8000)
        MFCC(numcep=26, winlen=0.02, winstep=0.01, sample_rate=8000)
    """

    def __init__(
        self,
        numcep: int,
        winlen: float = 0.025,
        winstep: float = 0.02,
        sample_rate: int = 16000,
    ):
        self.numcep = numcep
        self.winlen = winlen
        self.winstep = winstep
        self.sample_rate = sample_rate

    def __call__(self, audiodata: torch.Tensor) -> torch.Tensor:
        """Returns the MFCC for ``audiodata``.

        Args:
            audiodata: The audio signal from which to compute features. Size
                should be ``(T)``. i.e. a one-dimensional
                :py:class:`torch.Tensor`.

        Returns:
            ``torch.Tensor`` with size ``(numcep, T')``.
        """
        mfcc = python_speech_features.mfcc(
            audiodata.numpy(),
            samplerate=self.sample_rate,
            winlen=self.winlen,
            winstep=self.winstep,
            numcep=self.numcep,
            nfilt=self.numcep,
        )
        return torch.tensor(
            mfcc.T,
            dtype=audiodata.dtype,
            device=audiodata.device,
            requires_grad=audiodata.requires_grad,
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(numcep={self.numcep}, "
            f"winlen={self.winlen}, "
            f"winstep={self.winstep}, "
            f"sample_rate={self.sample_rate})"
        )


class AddSequenceLength:
    """Adds sequence length information to ``data``.

    Args:
        length_dim: Index of sequence length dimension.

    Example:
        >>> add_seq_len = AddSequenceLength(length_dim=0)
        >>> x = torch.rand([5, 10])
        >>> x_prime, x_len = add_seq_len(x)
        >>> bool(torch.all(x_prime == x))
        True
        >>> bool(x_len == torch.tensor(5))
        True
    """

    def __init__(self, length_dim=0):
        self.length_dim = length_dim

    def __call__(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tuple of ``data`` and a tensor containing its length."""
        seq_len = data.size(self.length_dim)
        return data, torch.tensor([seq_len], requires_grad=False)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(length_dim={self.length_dim})"
