from myrtlespeech.run.callbacks.callback import Callback


class GradientAccumulation(Callback):
    """Performs gradient accumulation.

    Args:
        accumulation_steps: Number of steps to perform before each optimizer
            and scheduler step.
    """

    def __init__(
        self, accumulation_steps: int,
    ):
        super().__init__()
        self.accumulation_steps = accumulation_steps

    def on_backward_begin(self, **kwargs):
        """Scale loss if training."""
        if self.training:
            kwargs["last_loss"] /= self.accumulation_steps
        return kwargs

    def on_backward_end(self, **kwargs):
        """Skips accumulation step."""
        if (kwargs["epoch_minibatches"] + 1) % self.accumulation_steps != 0:
            kwargs["skip_step"] = True
        return kwargs

    def on_step_end(self, **kwargs):
        if (kwargs["epoch_minibatches"] + 1) % self.accumulation_steps != 0:
            kwargs["skip_zero"] = True
        return kwargs
