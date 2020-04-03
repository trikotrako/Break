from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('bert')
class BertPredictor(Predictor):
    """
    Predictor for sequence to sequence models with dynamic vocabulary, including
    :class:`~allennlp.models.encoder_decoder.simple_seq2seq_dynamic`
    """

    def predict(self, source: str, allowed_tokens: str) -> JsonDict:
        return self.predict_json({"source": source, "allowed_tokens": allowed_tokens})
