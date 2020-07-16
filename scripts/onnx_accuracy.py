import argparse

import google.protobuf.text_format as text_format
import onnxruntime as ort
import torch
from myrtlespeech.builders.task_config import build
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.callbacks.callback import CallbackHandler
from myrtlespeech.run.run import ReportCTCDecoder
from myrtlespeech.run.run import WordSegmentor

RNN_BIDIRECTIONAL = True
RNN_HIDDEN_SIZE = 2048
NUM_RNN_LAYERS = 1
DS1_INPUT_NAMES = ["input", "in_lens", "h_n_in", "c_n_in"]
MAX_SAMPLES = 100  # Number of samples to use (full dataset is slow to eval)


class Dataset:
    """Builds dataset using :py:class:`MyrtlespeechConfig`.

    Args:
        config_fp: Filepath to a string representation of a
            :py:class:`task_config_pb2.TaskConfig` protobuf object
            containing the config for the desired task. If ``config_fp = None``
            this class will attempt to get the filepath with
            ``os.getenv('CONFIG_FP')``.
    """

    def __init__(self, config_fp):
        self._config_fp = config_fp

        assert self._config_fp is not None

        with open(self._config_fp) as f:
            pb = task_config_pb2.TaskConfig()
            task_config = text_format.Merge(f.read(), pb)

        # Load PyTorch model
        self.seq_to_seq, _, _, self.loader = build(
            task_config, download_data=True
        )
        self.cb_handler = CallbackHandler(
            callbacks=[
                ReportCTCDecoder(
                    self.seq_to_seq.post_process,
                    self.seq_to_seq.alphabet,
                    word_segmentor=WordSegmentor(" "),
                ),
            ],
            training=False,
        )


class ORTSessionWrap(torch.nn.Module):
    """Wraps an onnxruntime session in a :py:class:`torch.nn.Module`.

    Args:
        session: ort session.


        return_time: Boolean. If :py:data:`True`, return time for ort
            inference.
    """

    def __init__(self, session):
        super().__init__()
        self.session = session

    def forward(self, x, hx=None):
        input, in_lens = x

        if hx is None:
            h_n, c_n = get_lstm_hidden(batch=len(in_lens))
        else:
            h_n, c_n = hx
        args = (input, in_lens, h_n, c_n)
        dict_args = torch_to_numpy_dict(args)

        outputs = self.session.run(None, dict_args)

        outputs = [torch.tensor(x) for x in outputs]
        out, out_lens, h_n, c_n = outputs

        return (out, out_lens), (h_n, c_n)


def onnx_wrap(path):
    return ORTSessionWrap(ort.InferenceSession(path))


def create_dataset(path):
    return Dataset(path)


def get_lstm_hidden(
    batch: int,
    bidirectional: int = RNN_BIDIRECTIONAL,
    num_layers: int = NUM_RNN_LAYERS,
    hidden_size: int = RNN_HIDDEN_SIZE,
):
    num_directions = 2 if bidirectional else 1
    zeros = torch.zeros(num_layers * num_directions, batch, hidden_size)
    return zeros, zeros


def torch_to_numpy_dict(args):
    args_np = [x.numpy() if isinstance(x, torch.Tensor) else x for x in args]
    return {k: args_np[idx] for idx, k in enumerate(DS1_INPUT_NAMES)}


def test_accuracy(run_model, dataset, max_samples=MAX_SAMPLES):
    decoder_name = dataset.seq_to_seq.post_process.__class__.__name__
    cb_handler = dataset.cb_handler

    cb_handler.on_train_begin(epochs=0)
    run_model.train(mode=False)
    cb_handler.train(mode=False)

    for idx, (x, y) in enumerate(dataset.loader):
        # model
        x, y = cb_handler.on_batch_begin(x, y)
        out, _ = run_model(x)

        # loss
        _, _ = cb_handler.on_loss_begin(out, y)

        if cb_handler.on_batch_end() or (
            max_samples is not None and idx >= max_samples
        ):
            break

    cb_handler.on_epoch_end()

    wer = cb_handler.state_dict["reports"][decoder_name]["wer"]
    return {"wer": wer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ONNX model")
    parser.add_argument(
        "model", type=str, help="Location of ONNX model to use",
    )
    parser.add_argument("config", type=str, help="DS1 config file")
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=MAX_SAMPLES,
        help="Number of samples to use",
    )
    args = parser.parse_args()

    wer = test_accuracy(
        onnx_wrap(args.model), create_dataset(args.config), args.num_samples,
    )["wer"]
    print(f"WER: {wer:.3f}")
