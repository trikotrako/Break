from typing import Dict, List
import collections
import logging
import math

import torch
from overrides import overrides
from transformers import BertModel, BertConfig, BertTokenizer, PreTrainedTokenizer

from allennlp.common import JsonDict
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import sequence_cross_entropy_with_logits, get_text_field_mask


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


BERT_LARGE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                     "hidden_act": "gelu",
                     "hidden_dropout_prob": 0.1,
                     "hidden_size": 1024,
                     "initializer_range": 0.02,
                     "intermediate_size": 4096,
                     "max_position_embeddings": 512,
                     "num_attention_heads": 16,
                     "num_hidden_layers": 24,
                     "type_vocab_size": 2,
                     "vocab_size": 30522
                    }

BERT_BASE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "type_vocab_size": 2,
                    "vocab_size": 30522
                   }


@Model.register('simple_bert')
class BertForQuestionAnswering(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model_type: str,
                 pretrained_archive_path: str,
                 null_score_difference_threshold: float = 0.0,
                 model_is_for_squad1: bool = False,
                 n_best_size: int = 20,
                 max_answer_length: int = 30) -> None:
        super().__init__(vocab)
        if bert_model_type == "bert_base":
            config_to_use = BERT_BASE_CONFIG
        elif bert_model_type == "bert_large":
            config_to_use = BERT_LARGE_CONFIG
        else:
            raise RuntimeError(f"`bert_model_type` should either be \"bert_large\" or \"bert_base\"")
        config = BertConfig(vocab_size_or_config_json_file=config_to_use["vocab_size"],
                            hidden_size=config_to_use["hidden_size"],
                            num_hidden_layers=config_to_use["num_hidden_layers"],
                            num_attention_heads=config_to_use["num_attention_heads"],
                            intermediate_size=config_to_use["intermediate_size"],
                            hidden_act=config_to_use["hidden_act"],
                            hidden_dropout_prob=config_to_use["hidden_dropout_prob"],
                            attention_probs_dropout_prob=config_to_use["attention_probs_dropout_prob"],
                            max_position_embeddings=config_to_use["max_position_embeddings"],
                            type_vocab_size=config_to_use["type_vocab_size"],
                            initializer_range=config_to_use["initializer_range"])
        self.bert_qa_model = BertModel(config)
        self._loaded_qa_weights = False
        self._pretrained_archive_path = pretrained_archive_path
        self._null_score_difference_threshold = null_score_difference_threshold
        self._model_is_for_squad1 = model_is_for_squad1
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        start_logits, end_logits = self.bert_qa_model.forward(source_tokens['tokens'])
        output = {"tag_logits": start_logits}

        if target_tokens is not None:
            target_mask = torch.FloatTensor(torch.ones_like(target_tokens['tokens']).float())
            targets = target_tokens['tokens']
            relevant_targets = targets[:, 1:].contiguous()

            # shape: (batch_size, num_decoding_steps)
            relevant_mask = target_mask[:, 1:].contiguous()
            output["loss"] = sequence_cross_entropy_with_logits(start_logits, relevant_targets, relevant_mask)  # TODO

        return output
