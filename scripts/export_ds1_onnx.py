"""Exports a trained :py:class:`DeepSpeech1` model to ONNX for use with other
machine learning frameworks
"""
import argparse
from typing import Tuple

import google.protobuf.text_format as text_format
import torch
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.model.deep_speech_1 import DeepSpeech1
from myrtlespeech.model.rnn import init_hidden_state
from myrtlespeech.protos import task_config_pb2


def export_ds1(
    ds1_cfg_fp: str,
    weights_fp: str,
    onnx_fp: str,
    opset_version: int = 11,
    rename_lengths: bool = False,
):
    """Exports :py:class:`DeepSpeech1` model.

    Args:
        ds1_cfg_fp: filepath to config for DeepSpeech1 task config.

        weights_fp: filpath of weights.

        onnx_fp: filepath to save onnx file to.

        opset_version: onnx opset_version.

        rename_lengths: User should pass :py:Data`True` if the onnx compiler
            renames the ``lens`` inputs/outputs after compiling. To access
            this information, print
            ``[x.name for x in ort_session.get_inputs()]`` (or equivalent
            for the graph outputs). This is necessary because, for some
            combinations of opset and hard/soft-LSTM, the onnx exported
            correctly realises that sequence lengths are unchanged from input
            to output whereas in other cases it does not. In the latter case
            it renames the input or output names (e.g.``lens -> lens.1``)
            which breaks the ort session.
    """
    # Define attributes for ONNX export
    input_names = ["input", "lens", "h_n_in", "c_n_in"]
    dynamic_axes = {
        "input": {0: "batch", 3: "seq_len"},
        "lens": {0: "batch"},
        "h_n_in": {1: "batch"},
        "c_n_in": {1: "batch"},
        "output": {0: "seq_len", 1: "batch"},
        "c_n_out": {1: "batch"},
    }
    if rename_lengths:
        output_names = ["output", "lens_out", "h_n_out", "c_n_out"]
        dynamic_axes["lens_out"] = {0: "batch"}
    else:
        output_names = ["output", "lens", "h_n_out", "c_n_out"]

    # build model and load weights
    with open(ds1_cfg_fp) as f:
        pb = task_config_pb2.TaskConfig()
        task_config = text_format.Merge(f.read(), pb)
    stt = build_stt(task_config.speech_to_text)
    ds1 = stt.model
    state_dict = torch.load(weights_fp, map_location=torch.device("cpu"))
    ds1.load_state_dict(state_dict=state_dict, strict=True)

    # gen ds1 input args
    args = gen_ds1_args(ds1)

    model = CollapseArgsDS1(ds1)

    model.eval()
    example_outputs = model(*args)

    # trace model
    model = torch.jit.trace(model, args)

    torch.onnx.export(
        model,
        args,
        onnx_fp,
        export_params=True,
        verbose=False,
        example_outputs=example_outputs,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
    )


def gen_ds1_args(model: DeepSpeech1, id: int = 0, batch: int = 2) -> Tuple:
    """Returns valid input args for :py:class:`DeepSpeech1`.

    When  ``batch > 1``, this assumes that the first input dimension is
    the batch size.

    Args:
        id: Unique identifier used in generation (or retrieval) of model
            input args. This might be a dataset index or a seed for random
            generation.

        batch: Batch size of generated args.

    Returns:
        A Tuple ``x, xlen, hn, cn``` where ``x`` and ``xlens`` are
        respectively, the audio input and associated lengths and ``hn`` and
        ``cn`` are respectively, the LSTM hidden state and cell state.
    """
    torch.manual_seed(id)
    max_len = 20
    seq_len = torch.randint(1, max_len, (1,)).item()
    x = torch.randn(batch, model.input_channels, model.input_features, seq_len)
    if seq_len == 1:
        xlen = torch.ones((batch,), dtype=torch.int64)
    else:
        xlen = torch.randint(
            low=1, high=seq_len, size=(batch,), dtype=torch.int64
        )
        xlen = xlen.sort(descending=True)[0]
        xlen[0] = seq_len
    hn, cn = init_hidden_state(  # type: ignore
        batch=batch,
        dtype=x.dtype,
        hidden_size=model.bi_lstm.rnn.hidden_size,
        num_layers=model.bi_lstm.rnn.num_layers,
        bidirectional=model.bi_lstm.rnn.bidirectional,
        rnn_type=0,
    )
    return x, xlen, hn, cn


class CollapseArgsDS1(torch.nn.Module):
    """Collapses input args and flattens output args.

    This is necessary as the ``dynamic_axes`` arg for
    :py:meth:`torch.onnx.export` doesn't accept nested Tuples.

    Args:
        model: A :py:class:`DeepSpeech1`.
    """

    def __init__(self, model: DeepSpeech1):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        lens: torch.Tensor,
        h_n: torch.Tensor,
        c_n: torch.Tensor,
    ):
        (out, out_lens), (h_n, c_n) = self.model((x, lens), (h_n, c_n))
        return out, out_lens, h_n, c_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export DeepSpeech1 model ONNX graph"
    )
    parser.add_argument("config", help="The ds1 config file.")
    parser.add_argument("weights_fp", help="The filepath of trained weights.")
    parser.add_argument(
        "onnx_fp", help="The path the onnx model should be written to."
    )
    parser.add_argument(
        "opset_version",
        help="The onnx opset version to use for the export.",
        type=int,
    )
    parser.add_argument(
        "-r",
        "--rename",
        help=(
            "If flag is passed, onnx export renames input/output sequence "
            "length arguments."
        ),
        action="store_const",
        const=True,
    )
    args = parser.parse_args()
    export_ds1(
        args.config,
        args.weights_fp,
        args.onnx_fp,
        args.opset_version,
        args.rename,
    )
