
import json
import os

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.models.model import Model

from evaluation.decomposition import Decomposition, get_decomposition_from_tokens
from model.model_base import ModelBase

from dataset_readers.simple_seq2seq_dynamic import Seq2SeqDynamicDatasetReader
from model.seq2seq.simple_seq2seq_dynamic import SimpleSeq2SeqDynamic
from model.seq2seq.simple_seq2seq_dynamic_predictor import Seq2SeqDynamicPredictor

from dataset_readers.bert_model_reader import SquadReaderForPretrainedBert
from transformers import BertTokenizer, BertModel


class BertModelQDMR(ModelBase):
    def __init__(self, model_dir, cuda_device=-1):
        super(BertModelQDMR, self).__init__()

        self.model_dir = model_dir

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.predictor = Predictor(self.model, SquadReaderForPretrainedBert())

    def _decompose(self, question, verbose):
        pred = self.predictor.predict_json({"source": question})

        if self.beam:
            decomposition = get_decomposition_from_tokens(pred["predicted_tokens"][0])
        else:
            decomposition = get_decomposition_from_tokens(pred["predicted_tokens"])

        return Decomposition(decomposition)

    def predict(self, questions, print_non_decomposed, verbose, extra_args=None):
        # extra_args is used here to pass allowed tokens for the dynamic seq2seq model.
        if extra_args:
            inputs = [{"source": question, "allowed_tokens": allowed_tokens}
                      for question, allowed_tokens in zip(questions, extra_args)]
        else:
            inputs = [{"source": question} for question in questions]
        preds = self.predictor.predict_batch_json(inputs)

        return self._get_decompositions_from_predictions(preds)

    def _get_decompositions_from_predictions(self, preds):
        if self.beam:
            decompositions = [
                get_decomposition_from_tokens(pred["predicted_tokens"][0])
                for pred in preds
            ]
        else:
            decompositions = [
                get_decomposition_from_tokens(pred["predicted_tokens"])
                for pred in preds
            ]

        return decompositions

    def load_decompositions_from_file(self, predictions_file):
        with open(predictions_file, "r") as fd:
            preds = [json.loads(line) for line in fd.readlines()]

        return self._get_decompositions_from_predictions(preds)

