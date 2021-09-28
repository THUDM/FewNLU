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


import random
import string
from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import torch
from transformers import GPT2Tokenizer

import log
from utils import InputExample, get_verbalization_ids, PatternExample

logger = log.get_logger()
PVPOutputPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    Abstract class that provides different ways of organizing inputs.
    """

    def __init__(self, tokenizer, max_seq_length, label_list, use_cloze,
                 use_continuous_prompt, pattern_id, seed):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.use_cloze = use_cloze
        self.use_continuous_prompt = use_continuous_prompt
        self.pattern_id = pattern_id

        # self.is_multi_token = is_multi_token
        self._is_multi_token = None
        self.rng = random.Random(seed)

        if (not self.is_multi_token) and self.use_cloze:
            self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    @property
    def is_multi_token(self):
        return self._is_multi_token

    @property
    def prompt_length(self) -> int:
        """Return the number of continuous prompt tokens."""
        if self.use_cloze and self.use_continuous_prompt:
            return self.pattern_id
        else:
            return 0

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        if not self.is_multi_token:
            return max(len(self.verbalize(label)) for label in self.label_list)
        else:
            raise ValueError("Not supported for multi-token tasks.")


    @abstractmethod
    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a data_utils1 requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    @abstractmethod
    def available_patterns(self):
        """
        Return all available pattern ids.
        :return:
        """
        pass

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]


    def _build_mlm_logits_to_cls_logits_tensor(self):
        m2c_tensor = torch.ones([len(self.label_list), self.max_num_verbalizers], dtype=torch.long) * -1
        for label_idx, label in enumerate(self.label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor


    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        mask_count = input_ids.count(self.mask_id)
        labels = [-1] * len(input_ids)
        if mask_count == 0:
            return labels
        if mask_count == 1:
            assert (not self.is_multi_token)
        label_idx = input_ids.index(self.mask_id)
        for idx in range(label_idx, label_idx + mask_count):
            labels[idx] = 1
        return labels


    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            # return parts_a, parts_b
            return

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)


    def _encode_single_example(self, example: InputExample, labeled:bool):

        prompt_id = len(self.tokenizer.get_vocab())
        kwargs = {'add_prefix_space': True} if isinstance(self.tokenizer, GPT2Tokenizer) else {}
        raw_parts_a, raw_parts_b = self.get_parts(example)
        # logger.info('text_a: {}, text_b: {}'.format(raw_parts_a,raw_parts_b))
        def encoded_input(raw_parts):
            parts, block_flags = [], []
            for (x, s) in raw_parts:
                if isinstance(x, str):
                    out = self.tokenizer.encode(x, add_special_tokens=False, **kwargs)
                    flag = [0] * len(out)
                elif isinstance(x, int):
                    out = [prompt_id] * x
                    flag = [-1] * x
                else:
                    out = x
                    flag = [0] * len(x)
                parts.append((out, s))
                block_flags.append((flag, s))
            return parts, block_flags

        raw_parts_a = [x if isinstance(x, tuple) else (x, False) for x in raw_parts_a]
        parts_a, flags_a = encoded_input(raw_parts_a)
        parts_b, flags_b = [], []
        if raw_parts_b:
            raw_parts_b = [x if isinstance(x, tuple) else (x, False) for x in raw_parts_b]
            parts_b, flags_b = encoded_input(raw_parts_b)

        self.truncate(parts_a, parts_b, max_length=self.max_seq_length)
        self.truncate(flags_a, flags_b, max_length=self.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        flags_a = [token_id for part, _ in flags_a for token_id in part]
        flags_b = [token_id for part, _ in flags_b for token_id in part] if flags_b else None

        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        block_flags = self.tokenizer.build_inputs_with_special_tokens(flags_a, flags_b)

        if self.use_cloze and labeled:
            assert input_ids.count(self.mask_id) == 1, "Only for single-token task"
            mask_idx = input_ids.index(self.mask_id)
            assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
            verbalizer = self.verbalize(example.label)
            assert len(verbalizer) == 1, 'priming only supports one verbalization per label'
            verbalizer = verbalizer[0]
            verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
            input_ids[mask_idx] = verbalizer_id

        return input_ids, token_type_ids, block_flags

    def encode(self, example: InputExample, priming: bool=False) -> PatternExample:
        if priming:
            input_ids, token_type_ids, block_flags = self._encode_single_example(example, labeled=False)
            priming_data = example.meta['priming_data']

            priming_input_ids = []
            priming_block_flags = []
            for priming_example in priming_data:
                pe_input_ids, _, pe_block_flags = self._encode_single_example(priming_example, labeled=True)
                priming_input_ids += pe_input_ids
                priming_block_flags += pe_block_flags

            input_ids = priming_input_ids + input_ids
            block_flags = priming_block_flags + block_flags

            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(input_ids)
            block_flags = self.tokenizer.build_inputs_with_special_tokens(block_flags)

        else:
            input_ids, token_type_ids, block_flags = self._encode_single_example(example, labeled=False)

        mlm_labels = self.get_mask_positions(input_ids)

        return PatternExample(guid=example.guid,
                              input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              block_flags=block_flags,
                              mlm_labels=mlm_labels,
                              label=example.label,
                              idx=example.idx,
                              logits=example.logits)




    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def convert_plm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        logits = torch.squeeze(logits, 1)  # remove second dimension as we always have exactly one <mask> per example
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits



