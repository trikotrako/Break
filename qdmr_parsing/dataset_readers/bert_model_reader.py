import csv
import json
import logging
import collections
from typing import List

import torch
from overrides import overrides
from pytorch_pretrained_bert import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


START_SYMBOL = "[CLS]"
END_SYMBOL = "[SEP]"

@DatasetReader.register("simple_bert")
class SquadReaderForPretrainedBert(DatasetReader):
    def __init__(self,
                 pretrained_bert_model_file: str,
                 lazy: bool = False,
                 max_query_length: int = 64,
                 max_sequence_length: int = 384,
                 document_stride: int = 128) -> None:
        super().__init__(lazy)
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model_file)
        self._source_tokenizer = self._tokenizer
        self._target_tokenizer = self._source_tokenizer
        self._max_query_length = max_query_length
        self._max_sequence_length = max_sequence_length
        self._document_stride = document_stride
        self._source_token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = self._source_token_indexers
        self._source_add_start_token = True
        self._delimiter = "\t"
        self._source_max_tokens = None
        self._target_max_tokens = None
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

    @overrides
    def _read(self, file_path):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if line_num > 100:
                    break  # TODO: remove
                if len(row) != 2:
                    raise RuntimeError("Invalid line format: %s (line number %d)" % (row, line_num + 1))
                source_sequence, target_sequence = row
                yield self.text_to_instance(source_sequence, target_sequence)
        if self._source_max_tokens and self._source_max_exceeded:
            logger.info("In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                        self._source_max_exceeded, self._source_max_tokens)
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info("In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                        self._target_max_exceeded, self._target_max_tokens)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        if self._source_add_start_token:
            source_string = START_SYMBOL + f" {source_string}"
        source_string = source_string + f" {END_SYMBOL}"

        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[:self._source_max_tokens]

        source_field = TextField([Token(w) for w in tokenized_source], self._source_token_indexers)
        if target_string is not None:
            target_string = START_SYMBOL + f" {target_string}"
            target_string = target_string + f" {END_SYMBOL}"
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[:self._target_max_tokens]
            target_field = TextField([Token(w) for w in tokenized_target], self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})
