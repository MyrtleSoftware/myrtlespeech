"""
Utilities for preprocessing audio data.
"""
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

    def __call__(self, audiodata: torch.Tensor):
        """Returns the MFCC for ``audiodata``.

        Args:
            audiodata: The audio signal from which to compute features. Size
                should be ``(T)``. i.e. a one-dimensional
                :py:class:`torch.Tensor`.

        Returns:
            ``torch.Tensor`` with size ``(T', numcep)``.
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
            mfcc,
            dtype=audiodata.dtype,
            device=audiodata.device,
            requires_grad=audiodata.requires_grad,
        )

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(numcep={self.numcep}, "
            f"winlen={self.winlen}, "
            f"winstep={self.winstep}, "
            f"sample_rate={self.sample_rate})"
        )
