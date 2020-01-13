from myrtlespeech.run.callbacks.callback import Callback


class GradientAccumulation(Callback):
    """Performs gradient accumulation.

    Args:
        model: See :py:class:`ModelCallback`.
    """

    def __init__(
        self, accumulation_steps: int,
    ):
        super().__init__()
        self.accumulation_steps = accumulation_steps
