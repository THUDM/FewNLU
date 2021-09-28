import copy
import os
import random
from typing import List

import numpy as np

import log
from utils import InputExample, softmax, LogitsList, eq_div

logger = log.get_logger()


def generate_ipet_train_sets(dataprovider, unlabeled_data: List[InputExample], labels: List[str],
                             logits_dir: str, output_dir: str, reduction: str, num_new_examples: int,
                             logits_percentage: float, n_most_likely: int = -1, seed: int = 42, logits_prefix='unlabeled', use_brother_fold_logits=False):
    """
    Generate training sets for the next generation of iPET models.
    :param train_data: the training examples
    :param unlabeled_data: the unlabeled examples
    :param labels: the list of all possible labels
    :param logits_dir: the directory that contains the predictions of all models in the current generation for the
           unlabeled data.
    :param output_dir: the output directory
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :param num_new_examples: the number of new examples to create
    :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
    :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
    :param seed: the random seed to use
    """
    pattern_dirs=[x for x in next(os.walk(logits_dir))[1] if x.startswith('p')]
    # other_pattern_logits_dirs=[]
    # parent_logits_dir=os.path.abspath(os.path.join(logits_dir, ".."))
    # cur_path_name=logits_dir.split('/')[-1]
    # for other_pattern_subdir in next(os.walk(parent_logits_dir))[1]:
    #     if other_pattern_subdir != cur_path_name: other_pattern_logits_dirs.append(os.path.join(parent_logits_dir,other_pattern_subdir))

    # if use_brother_folder_logits==True:
    #     parent_logits_dir=os.path.abspath(os.path.join(logits_dir, ".."))
    #     cur_folder_name=logits_dir.split('/')[-1]
    #     if cur_folder_name.startswith('fold'):
    #         for brother_subdir in next(os.walk(parent_logits_dir))[1]:
    #             if brother_subdir.startswith('fold') and brother_subdir!=cur_folder_name: brother_logits_dirs.append(os.path.join(parent_logits_dir,brother_subdir))
    #     else:
    #         grand_logits_dir=os.path.abspath(os.path.join(parent_logits_dir, ".."))
    #         parent_folder_name=parent_logits_dir.split('/')[-1]
    #         if parent_folder_name.startswith('fold'):
    #             for brother_subdir in next(os.walk(grand_logits_dir))[1]:
    #                 if brother_subdir.startswith('fold') and brother_subdir!=parent_folder_name: brother_logits_dirs.append(os.path.join(grand_logits_dir,brother_subdir,cur_folder_name))
    #         else:
    #             raise RuntimeError("The parent path and grandparent path are not started with 'fold', but use_brother_folder_logits==True")

    if dataprovider:
        examples_per_labels=[]
        for fold_id in range(dataprovider.fold_num):
            train_data, _ =dataprovider.get_splited_data(fold_id)
            train_examples_per_label = [sum(1 for ex in train_data if ex.label == label) for label in labels]
            multiplier = num_new_examples / len(train_data)
            examples_per_label = [int(epl * multiplier) for epl in train_examples_per_label]
            examples_per_labels.append(examples_per_label)
            logger.info(f"Example distribution in the original dataset: {train_examples_per_label}")
    else:
        examples_per_label = eq_div(num_new_examples, len(labels))

    logger.info(f"Target distribution for the new dataset: {examples_per_label}")

    for example in unlabeled_data:
        example.label, example.logits = None, None

    rng = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    def prepare_logits(logits_dir):
        subdirs = next(os.walk(logits_dir))[1]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

        logits_lists = {}

        for subdir in subdirs:
            results_file = os.path.join(logits_dir, subdir, 'results.txt')
            logits_file = os.path.join(logits_dir, subdir, '{}_logits.txt'.format(logits_prefix))
            logits = []

            if not os.path.exists(results_file) or not os.path.exists(logits_file):
                logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
                continue

            if reduction == 'mean':
                result_train = 1
            else:
                with open(results_file, 'r') as fh:
                    results = ast.literal_eval(fh.read())
                    result_train = results['train_set_before_training']

            with open(logits_file, 'r') as fh:
                for line in fh.read().splitlines():
                    example_logits = [float(x) for x in line.split()]
                    logits.append(example_logits)

            logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
                results_file, result_train, len(logits), len(logits[0])))

            loglist = LogitsList(score=result_train, logits=logits)
            logits_lists[subdir] = loglist
        return subdirs,logits_lists
    
    # subdirs,logits_lists=prepare_logits(logits_dir)
    # other_pattern_logits=[]
    # if len(other_pattern_logits_dirs)!=0:
    #     for other_pattern_logits_dir in other_pattern_logits_dirs:
    #         tmp_dirs,tmp_logits=prepare_logits(other_pattern_logits_dir)
    #         other_pattern_logits.append(tmp_logits)
    subdirs={};logits_lists={}
    for pattern_dir in pattern_dirs:
        pattern_name=pattern_dir.split('/')[-1]
        tmp_dirs,tmp_logits_lists=prepare_logits(os.path.join(logits_dir,pattern_dir))
        subdirs[pattern_name]=tmp_dirs
        logits_lists[pattern_name]=tmp_logits_lists

    for (pattern_name,p_subdirs) in subdirs.items():
        for subdir in p_subdirs:
            fold=int(subdir.split('-')[0][1:])
            examples_per_label = examples_per_labels[fold] if examples_per_labels is not None else examples_per_label
            other_logits_lists=[]
            if use_brother_fold_logits==False:
                for (pattern,tmp_logits) in logits_lists.items():
                    if pattern==pattern_name:
                        other_logits_lists+=[ll for sd, ll in tmp_logits.items() if sd != subdir and sd.startswith(subdir.split('-')[0])]
                    else:
                        other_logits_lists+=[ll for (sd,ll) in tmp_logits.items() if sd.startswith(subdir.split('-')[0])]
            else:
                for (pattern,tmp_logits) in logits_lists.items():
                    if pattern==pattern_name:
                        other_logits_lists+=[ll for sd, ll in tmp_logits.items() if sd != subdir]
                    else:
                        other_logits_lists+=[ll for (sd,ll) in tmp_logits.items()]

            subdir_train_set = generate_ipet_train_set(
                other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
                logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng,
                rng_np=rng_np
            )

            InputExample.save_examples(subdir_train_set,
                                    os.path.join(output_dir, pattern_name+'-'+subdir + '-' + logits_prefix + '-train.bin')) #TODO


def generate_ipet_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                            examples_per_label: List[int], logits_percentage: float, reduction: str = 'mean',
                            n_most_likely: int = -1, rng=None, rng_np=None) -> List[InputExample]:
    """
    Generate a single training set for the next generation of iPET models.
    :param logits_lists: predictions from the previous generation of models
    :param labels: all task labels
    :param original_data: the original training data corresponding to the logits_lists
    :param examples_per_label: the number of examples per label to create
    :param logits_percentage: the percentage of models/logits to choose
    :param reduction: the reduction strategy ('wmean' or 'mean')
    :param n_most_likely: if >0, for each label the n_most_likely examples with the highest logits are chosen
    :param rng: the random number generator to use for non-numpy operations
    :param rng_np: the random number generator to use for numpy operations
    :return: a list of input examples that serves as training set for the next generation
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1

    if not rng:
        rng = random.Random()
    if not rng_np:
        rng_np = np.random.RandomState()

    num_logits_lists = max(round(len(logits_lists) * logits_percentage),1) #TODO
    logits_lists = rng.sample(logits_lists, k=num_logits_lists)
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0)
        logits = softmax(logits, axis=1).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
        logits = softmax(logits, axis=1).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]

    test_set = []
    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info("There are {} examples for label {}".format(len(examples), label))
            while len(examples) < examples_per_label[idx] and len(examples)!=0: #TODO
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [copy.deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples, num_examples=examples_per_label[idx], rng=rng_np)
        test_set.extend(label_examples)

    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int, rng) -> List[InputExample]:
    if len(examples)==0: return examples #TODO
    label_probabilities = [max(example.logits) for example in examples]
    sum_label_probabilities = sum(label_probabilities)
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    return rng.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()
