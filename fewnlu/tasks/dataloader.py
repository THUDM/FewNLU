import random
from collections import Counter, defaultdict
from typing import List

import log as log
from tasks.superglue.processor import SUPERGLUE_PROCESSORS
from tasks.superglue.pvp import SUPERGLUE_PVPS, SUPERGLUE_METRICS
from utils import InputExample, eq_div
from tasks.base_processor import ProcessorOutputPattern

from global_vars import UNLABELED_SET, TRAIN_SET, DEV32_SET, DEV_SET, TEST_SET, AUGMENTED_SET, SET_TYPES

logger = log.get_logger()

def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
        logger.info("shuffle and restrict data examples.")
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples



def load_examples(dataset_name: str,
                  task_name: str,
                  data_dir: str,
                  set_type: str,
                  use_cloze: bool,
                  num_examples: int,
                  num_examples_per_label: int,
                  seed: int) -> ProcessorOutputPattern:
    """Load data examples for a given task processor."""

    assert (num_examples is not None) or (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None),  "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = DATASETS[dataset_name]["processors"][task_name](task_name)

    ex_str = f"num_examples={num_examples}" if num_examples is not None else f"num_examples_per_label={num_examples_per_label}"
    logger.info(f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})")

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == DEV32_SET:
        examples = processor.get_dev32_examples(data_dir, use_cloze)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir, use_cloze)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    elif set_type == AUGMENTED_SET:
        examples = processor.get_augmented_examples(data_dir)
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    # both are not used
    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        assert limited_examples.is_full() == True
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples





def load_dataset(args):

    use_cloze = args.use_cloze
    dataset_name = args.dataset_name
    task_name = args.task_name
    data_dir = args.data_dir
    seed = args.seed

    train_ex_per_label, dev32_ex_per_label = None, None
    train_ex, dev32_ex = args.train_examples, args.dev32_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        dev32_ex_per_label = eq_div(args.dev32_examples, len(args.label_list)) if args.dev32_examples != -1 else -1

    train_data = load_examples(dataset_name, task_name, data_dir, TRAIN_SET, use_cloze, num_examples=train_ex, num_examples_per_label=train_ex_per_label, seed=seed)
    dev32_data = load_examples(dataset_name, task_name, data_dir, DEV32_SET, use_cloze,num_examples=dev32_ex, num_examples_per_label=dev32_ex_per_label, seed=seed)

    eval_ex_per_label = None
    if args.eval_set == 'test':
        eval_ex = args.test_examples
        eval_set = TEST_SET
    else:
        eval_ex = args.dev_examples
        eval_set = DEV_SET
    if args.split_examples_evenly:
        eval_ex_per_label = eq_div(eval_ex, len(args.label_list)) if eval_ex != -1 else -1

    eval_data = load_examples(dataset_name, task_name, data_dir, eval_set, use_cloze, eval_ex, eval_ex_per_label, seed)

    # if (args.method == "lm_training" and args.use_unlabeled_data_lm_training) or (args.method == "ipet"):
    # import pdb 
    # pdb.set_trace()
    unlabeled_data = load_examples(dataset_name, task_name, data_dir, UNLABELED_SET, use_cloze, args.unlabeled_examples, None, seed)

    return train_data, dev32_data, eval_data, unlabeled_data

DATASETS={
    "superglue": {
        "processors": SUPERGLUE_PROCESSORS,
        "pvps": SUPERGLUE_PVPS,
        "metrics": SUPERGLUE_METRICS,
    }
}