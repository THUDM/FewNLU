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

"""
This file contains the logic for loading training and test data for all SuperGLUE tasks.
"""

import json
import os
import random
from collections import Counter
from typing import List, Dict, Callable

import log
from utils import InputExample
from global_vars import AUGMENTED_SET, TRAIN_SET, DEV32_SET, DEV_SET, TEST_SET, UNLABELED_SET
from tasks.base_processor import DataProcessor

logger = log.get_logger()

class SuperGLUEDataProcessor(DataProcessor):
    """
    Data processsor for SuperGLUE tasks.
    """

    TRAIN_FILE = "train.jsonl"
    DEV_FILE = "val.jsonl"
    DEV32_FILE = "dev32.jsonl"
    TEST_FILE = "test.jsonl"
    UNLABELED_FILE = "unlabeled.jsonl"
    AUGMENTED_FILE = "augmented.jsonl"

    WSC_TRAIN_FILE_FOR_CLS = "train_for_cls.jsonl"
    WSC_DEV32_FILE_FOR_CLS = "dev32_for_cls.jsonl"

    def __init__(self, task_name: str):
        super(SuperGLUEDataProcessor, self).__init__()
        self.task_name = task_name
        assert self.task_name in SUPERGLUE_PROCESSORS

    def get_train_examples(self, data_dir, use_cloze):
        if not use_cloze and self.task_name == "wsc":
            logger.info("Loading CLS train set for WSC task.")
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_TRAIN_FILE_FOR_CLS),  TRAIN_SET, use_cloze=False)
        elif self.task_name=='wsc':
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_TRAIN_FILE_FOR_CLS),  TRAIN_SET, use_cloze=True)
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.TRAIN_FILE), TRAIN_SET)

    def get_dev32_examples(self, data_dir, use_cloze):
        if not use_cloze and self.task_name == "wsc":
            logger.info("Loading CLS dev32 set for WSC task.")
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_DEV32_FILE_FOR_CLS),  TRAIN_SET, use_cloze=False)
        elif self.task_name=='wsc':
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_DEV32_FILE_FOR_CLS),  TRAIN_SET, use_cloze=True)
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.DEV32_FILE), DEV32_SET)

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.DEV_FILE), DEV_SET)

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.TEST_FILE), TEST_SET)

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.UNLABELED_FILE), UNLABELED_SET)

    def get_augmented_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.AUGMENTED_FILE), AUGMENTED_SET)


class RteProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE RTE task."""

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, path: str, set_type: str, hypothesis_name: str = "hypothesis", premise_name: str = "premise") -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = line_idx
                label = example_json.get('label')
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json[premise_name]
                text_b = example_json[hypothesis_name]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
        return examples


class CbProcessor(RteProcessor):
    """Processor for the SuperGLUE CB task."""

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]


class WicProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE WiC task."""

    def get_labels(self):
        return ["F", "T"]

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    idx = int(idx)
                label = "T" if example_json.get('label') else "F"
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['sentence1']
                text_b = example_json['sentence2']
                meta = {'word': example_json['word']}
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx, meta=meta)
                examples.append(example)
        return examples


class WscProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE WSC task."""

    def get_labels(self):
        return ["False", "True"]

    def _create_examples(self, path: str, set_type: str, use_cloze: bool=True) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['text']
                meta = {
                    'span1_text': example_json['target']['span1_text'],
                    'span2_text': example_json['target']['span2_text'],
                    'span1_index': example_json['target']['span1_index'],
                    'span2_index': example_json['target']['span2_index']
                }

                # the indices in the dataset are wrong for some examples, so we manually fix them
                span1_index, span1_text = meta['span1_index'], meta['span1_text']
                span2_index, span2_text = meta['span2_index'], meta['span2_text']
                words_a = text_a.split()
                words_a_lower = text_a.lower().split()
                words_span1_text = span1_text.lower().split()
                span1_len = len(words_span1_text)

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    for offset in [-1, +1]:
                        if words_a_lower[span1_index + offset:span1_index + span1_len + offset] == words_span1_text:
                            span1_index += offset

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    logger.warning(f"Got '{words_a_lower[span1_index:span1_index + span1_len]}' but expected "
                                   f"'{words_span1_text}' at index {span1_index} for '{words_a}'")

                if words_a[span2_index] != span2_text:
                    for offset in [-1, +1]:
                        if words_a[span2_index + offset] == span2_text:
                            span2_index += offset

                    if words_a[span2_index] != span2_text and words_a[span2_index].startswith(span2_text):
                        words_a = words_a[:span2_index] \
                                  + [words_a[span2_index][:len(span2_text)], words_a[span2_index][len(span2_text):]] \
                                  + words_a[span2_index + 1:]

                assert words_a[span2_index] == span2_text, \
                    f"Got '{words_a[span2_index]}' but expected '{span2_text}' at index {span2_index} for '{words_a}'"

                text_a = ' '.join(words_a)
                meta['span1_index'], meta['span2_index'] = span1_index, span2_index

                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                # if use_cloze and (set_type == TRAIN_SET or set_type == DEV32_SET) and label != 'True':
                #     continue
                examples.append(example)
        return examples


class BoolQProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE BoolQ task."""

    def get_labels(self):
        return ["False", "True"]

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
        return examples


class CopaProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE COPA task."""

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = str(example_json['label']) if 'label' in example_json else None
                idx = example_json['idx']
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['premise']
                meta = {
                    'choice1': example_json['choice1'],
                    'choice2': example_json['choice2'],
                    'question': example_json['question']
                }
                example = InputExample(guid=guid+'-o', text_a=text_a, label=label, meta=meta, idx=idx)
                examples.append(example)

        if set_type == TRAIN_SET or set_type == UNLABELED_SET or set_type == DEV32_SET:
            mirror_examples = []
            for ex in examples:
                label = "1" if ex.label == "0" else "0"
                meta = {
                    'choice1': ex.meta['choice2'],
                    'choice2': ex.meta['choice1'],
                    'question': ex.meta['question']
                }
                mirror_example = InputExample(guid=ex.guid + 'm', text_a=ex.text_a, label=label, meta=meta)
                mirror_examples.append(mirror_example)
            examples += mirror_examples
            logger.info(f"Added {len(mirror_examples)} mirror examples, total size is {len(examples)}...")
        return examples


class MultiRcProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE MultiRC task."""

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json['idx']
                text = example_json['passage']['text']
                questions = example_json['passage']['questions']
                for question_json in questions:
                    question = question_json["question"]
                    question_idx = question_json['idx']
                    answers = question_json["answers"]
                    for answer_json in answers:
                        label = str(answer_json["label"]) if 'label' in answer_json else None
                        answer_idx = answer_json["idx"]
                        guid = f'{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}'
                        meta = {
                            'passage_idx': passage_idx,
                            'question_idx': question_idx,
                            'answer_idx': answer_idx,
                            'answer': answer_json["text"]
                        }
                        idx = [passage_idx, question_idx, answer_idx]
                        example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples


class RecordProcessor(SuperGLUEDataProcessor):
    """Processor for the SuperGLUE ReCoRD task."""

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, path, set_type, seed=42, max_train_candidates_per_question: int = 10) -> List[InputExample]:
        examples = []
        entity_shuffler = random.Random(seed)
        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                text = example_json['passage']['text']
                entities = set()
                for entity_json in example_json['passage']['entities']:
                    start = entity_json['start']
                    end = entity_json['end']
                    entity = text[start:end + 1]
                    entities.add(entity)

                entities = list(entities)
                text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations
                questions = example_json['qas']

                for question_json in questions:
                    question = question_json['query']
                    question_idx = question_json['idx']
                    answers = set()

                    for answer_json in question_json.get('answers', []):
                        answer = answer_json['text']
                        answers.add(answer)

                    answers = list(answers)

                    if set_type == TRAIN_SET:
                        # create a single example per *correct* answer
                        for answer_idx, answer in enumerate(answers):
                            candidates = [ent for ent in entities if ent not in answers]
                            if len(candidates) > max_train_candidates_per_question - 1:
                                entity_shuffler.shuffle(candidates)
                                candidates = candidates[:max_train_candidates_per_question - 1]

                            guid = f'{set_type}-p{idx}-q{question_idx}-a{answer_idx}'
                            meta = {
                                'passage_idx': idx,
                                'question_idx': question_idx,
                                'candidates': [answer] + candidates,
                                'answers': [answer]
                            }
                            ex_idx = [idx, question_idx, answer_idx]
                            example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta, idx=ex_idx)
                            examples.append(example)
                    else:
                        # create just one example with *all* correct answers and *all* answer candidates
                        guid = f'{set_type}-p{idx}-q{question_idx}'
                        meta = {
                            'passage_idx': idx,
                            'question_idx': question_idx,
                            'candidates': entities,
                            'answers': answers
                        }
                        example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples



"""
TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
}

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["f1", "em", "acc"],
    "record": ["em", "f1"]
}

DEFAULT_METRICS = ["acc"]
"""


SUPERGLUE_PROCESSORS = {
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor
}  # type: Dict[str,Callable[[],SuperGLUEDataProcessor]]

