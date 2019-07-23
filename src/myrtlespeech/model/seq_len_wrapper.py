import torch


class SeqLenWrapper(torch.nn.Module):
    """TODO

    """

    def __init__(self, module, seq_lens_fn):
        super().__init__()
        self.module = module
        self.seq_lens_fn = seq_lens_fn

    def forward(self, x, seq_lens=None, *args, **kwargs):
        result = self.module(x, *args, **kwargs)
        if seq_lens is None:
            return result
        return result, self.seq_lens_fn(seq_lens)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(module={self.module}, "
            + f"seq_lens_fn={self.seq_lens_fn})"
        )
