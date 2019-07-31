from typing import Tuple

from myrtlespeech.builders.cnn_rnn_encoder import build as build_cnn_rnn_encoder
from myrtlespeech.model.encoder.encoder import Encoder
from myrtlespeech.protos import encoder_pb2


def build(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: int = 1,
    seq_len_support: bool = False,
) -> Tuple[Encoder, int]:
    """Returns an :py:class:`.Encoder` based on the given config.

    Args:
        encoder_cfg: A :py:class:`myrtlespeech.protos.encoder_pb2.Encoder`
            protobuf object containing the config for the desired
            :py:class:`.Encoder`.

        input_features: The number of features for the input.

        input_channels: The number of channels for the input.

        seq_len_support: If :py:data:`True`, the returned encoder's
            :py:meth:`torch.nn.Module.forward` method must optionally accept a
            ``seq_lens`` kwarg. See :py:meth:`.Encoder.forward` for more
            information.

    Returns:
        A tuple containing an :py:class:`.Encoder` based on the config and the
        number of output features.

    Example:
        >>> from google.protobuf import text_format
        >>> encoder_cfg_text = '''
        ... cnn_rnn_encoder {
        ...   vgg {
        ...     vgg_config: A;
        ...     batch_norm: false;
        ...     use_output_from_block: 2;
        ...   }
        ...   rnn {
        ...     rnn_type: LSTM;
        ...     hidden_size: 1024;
        ...     num_layers: 5;
        ...     bias: true;
        ...     bidirectional: true;
        ...   }
        ... }
        ... '''
        >>> encoder_cfg = text_format.Merge(
        ...     encoder_cfg_text,
        ...     encoder_pb2.Encoder()
        ... )
        >>> build(encoder_cfg, input_features=10, input_channels=3)
        (CNNRNNEncoder(
          (cnn): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace)
            (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(inplace)
            (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (7): ReLU(inplace)
            (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (9): ReLU(inplace)
            (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (cnn_to_rnn): Lambda()
          (rnn): RNN(
            (rnn): LSTM(256, 1024, num_layers=5, bidirectional=True)
          )
        ), 2048)
    """
    encoder_choice = encoder_cfg.WhichOneof("supported_encoders")
    if encoder_choice == "cnn_rnn_encoder":
        return build_cnn_rnn_encoder(
            cnn_rnn_encoder_cfg=encoder_cfg.cnn_rnn_encoder,
            input_features=input_features,
            input_channels=input_channels,
            seq_len_support=seq_len_support,
        )
    else:
        raise ValueError(f"{encoder_choice} not supported")
