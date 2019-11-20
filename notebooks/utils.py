from myrtlespeech.run.callbacks.callback import Callback


class PrintCB(Callback):
    r"""Custom callback to monitor training and print results.

    This is unlikely to be useful outside of notebook setting.
    """

    def __init__(self):
        super().__init__()

    def on_batch_end(self, **kwargs):

        if self.training and kwargs["epoch_batches"] % 100 == 0:
            print(kwargs["epoch_batches"], kwargs["last_loss"].item())

            return
        epoch = kwargs["epoch"]
        if kwargs["epoch_batches"] % 100 == 0 and kwargs["epoch_batches"] != 0:
            print(f"{kwargs['epoch_batches']} batches completed")
            try:
                wer_reports = kwargs["reports"][
                    seq_to_seq.post_process.__class__.__name__
                ]
                wer = wer_reports["wer"]
                if len(wer_reports["transcripts"]) > 0:
                    transcripts = wer_reports["transcripts"][
                        0
                    ]  # take first element
                    pred, exp = transcripts
                    pred = "".join(pred)
                    exp = "".join(exp)
                    loss = kwargs["reports"]["ReportMeanBatchLoss"]
                    print(
                        "batch end, pred: {}, exp: {}, wer: {:.4f}".format(
                            pred, exp, wer
                        )
                    )

            except KeyError:
                print("no wer - using new decoder?")

    def on_epoch_end(self, **kwargs):
        if self.training:
            return
        epoch = kwargs["epoch"]

        try:

            loss = kwargs["reports"]["ReportMeanBatchLoss"]

            wer_reports = kwargs["reports"][
                seq_to_seq.post_process.__class__.__name__
            ]
            wer = wer_reports["wer"]

            out_str = "{}, loss: {:.8f}".format(epoch, loss)

            if len(wer_reports["transcripts"]) > 0:
                transcripts = wer_reports["transcripts"][
                    0
                ]  # take first element
                pred, exp = transcripts
                pred = "".join(pred)
                exp = "".join(exp)

                out_str += ", wer: {:.4f}, pred: {}, exp: {},".format(
                    wer, pred, exp
                )
            print(out_str)
        except KeyError:

            print("no wer - using new decoder?")
