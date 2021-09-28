# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from transformers import PreTrainedTokenizer

from tasks.dataloader import DATASETS
from utils import InputFeatures, InputExample, PLMInputFeatures
from global_vars import SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PLM_WRAPPER


class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_name:str, task_name: str, pattern_id: int,
                 use_cloze:bool, use_continuous_prompt:bool, max_seq_len:int, label_list: List, seed:int):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.pattern_id = pattern_id
        self.use_cloze = use_cloze
        self.use_continuous_prompt = use_continuous_prompt
        self.max_seq_len = max_seq_len
        self.label_list = label_list
        self.seed = seed

        PVPS = DATASETS[self.dataset_name]["pvps"]
        self.pvp = PVPS[self.task_name](self.tokenizer, self.max_seq_len, self.label_list, self.use_cloze,
                                        self.use_continuous_prompt, self.pattern_id, self.seed)

        self.label_map = {label: idx for idx, label in enumerate(self.label_list)}

    @abstractmethod
    def get_input_features(self, example:InputExample, labelled: bool, priming: bool, **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""
    def get_input_features(self, example:InputExample, labelled: bool, priming: bool, **kwargs) -> InputFeatures:
        pattern_example = self.pvp.encode(example, priming=priming)
        input_ids = pattern_example.input_ids
        token_type_ids = pattern_example.token_type_ids
        block_flags = pattern_example.block_flags
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_seq_len - len(input_ids)
        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        block_flags = block_flags + ([0] * padding_length)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len
        assert len(block_flags) == self.max_seq_len

        label_id = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
            """
            if self.wrapper.config.model_type == 'gpt2':
                # shift labels to the left by one
                mlm_labels.append(mlm_labels.pop(0))
            """
        else:
            mlm_labels = [-1] * self.max_seq_len

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label_id, mlm_labels=mlm_labels, logits=logits, idx=example.idx,
                             block_flags=block_flags)


class PLMPreprocessor(MLMPreprocessor):
    """Preprocessor for models pretrained using a permuted language modeling objective (e.g., XLNet)."""

    def get_input_features(self, example: InputExample, labelled: bool, priming: bool, **kwargs) -> PLMInputFeatures:
        input_features = super().get_input_features(example, labelled, priming, **kwargs)
        input_ids = input_features.input_ids

        num_masks = 1  # currently, PLMPreprocessor supports only replacements that require exactly one mask

        perm_mask = np.zeros((len(input_ids), len(input_ids)), dtype=np.float)
        label_idx = input_ids.index(self.pvp.mask_id)
        perm_mask[:, label_idx] = 1  # the masked token is not seen by any other token

        target_mapping = np.zeros((num_masks, len(input_ids)), dtype=np.float)
        target_mapping[0, label_idx] = 1.0

        return PLMInputFeatures(perm_mask=perm_mask, target_mapping=target_mapping, **input_features.__dict__)


class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""
    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        pattern_example = self.pvp.encode(example, priming=False)
        input_ids = pattern_example.input_ids
        token_type_ids = pattern_example.token_type_ids
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        mlm_labels = [-1] * len(input_ids)
        block_flags = [0] * len(input_ids)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len
        assert len(mlm_labels) == self.max_seq_len
        assert len(block_flags) == self.max_seq_len

        label = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        return InputFeatures(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             label=label,
                             mlm_labels=mlm_labels, logits=logits, idx=example.idx, block_flags=block_flags)



PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: SequenceClassifierPreprocessor,
    MLM_WRAPPER: MLMPreprocessor,
    PLM_WRAPPER: PLMPreprocessor,
}
