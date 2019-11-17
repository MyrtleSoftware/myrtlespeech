import torch.nn.functional as F

from myrtlespeech.model.rnn_t import label_collate


class TransducerDecoder:
    """Decoder base class.

    Args:
        alphabet: An Alphabet object.
        blank_symbol: The symbol in `alphabet` to use as the blank during CTC
            decoding.
        model: Model to use for prediction.
    """

    def __init__(self, blank_index, model):
        self._model = model
        self._SOS = -1   # start of sequence
        self._blank_id = blank_index

    def _pred_step(self, label, hidden, device):
        if label == self._SOS:
            return self._model.predict(None, hidden, add_sos=False)
        if label > self._blank_id:
            label -= 1
        label = label_collate([[label]]).to(device)
        return self._model.predict(label, hidden, add_sos=False)

    def _joint_step(self, enc, pred, log_normalize=False):
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels):
        return self._SOS if labels == [] else labels[-1]
