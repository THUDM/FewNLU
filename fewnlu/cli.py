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
This script can be used to train and evaluate either a few-shot method on
one of the supported tasks and datasets.
"""

import json
import shutil
import time
from collections import defaultdict
from typing import Dict
import statistics
import random
import itertools
import copy

from arguments import get_args
from configs import get_wrapper_config, get_train_eval_config, get_data_config
import os
import torch
import log
from utils import save_predictions, set_seed, save_logits, InputExample
from augmentation import generate_ipet_train_sets
from wrapper import TransformerModelWrapper
from tasks.dataloader import load_dataset, DATASETS

from global_vars import DEFAULT_METRICS, TRAIN_EVAL_CONFIG_NAME

logger = log.get_logger()

# def _write_results(path: str, results: Dict):
#     with open(path, 'w') as fh:
#         for metric in results.keys():
#             for pattern_id, values in results[metric].items():
#                 mean = statistics.mean(values)
#                 stdev = statistics.stdev(values) if len(values) > 1 else 0
#                 result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
#                 logger.info(result_str)
#                 fh.write(result_str + '\n')
#         for metric in results.keys():
#             all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
#             all_mean = statistics.mean(all_results)
#             all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
#             result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
#             logger.info(result_str)
#             fh.write(result_str + '\n')



def _write_results(path: str, results: Dict, dev32_results=None):

    ret_dict = {"dev32": {}, "dev": {}}
    with open(path, 'w') as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')
            ret_dict["dev"][metric] = all_mean
            # ret_dict["dev"]["all_stdev"][metric] = all_stdev

        if dev32_results is not None:
            for metric in dev32_results.keys():
                all_results = [result for pattern_results in dev32_results[metric].values() for result in pattern_results]
                all_mean = statistics.mean(all_results)
                all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
                result_str = "{}-dev32-all-p: {} +- {}".format(metric, all_mean, all_stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')
                ret_dict["dev32"][metric] = all_mean
                # ret_dict["dev32"]["all_stdev"][metric] = all_stdev
    return ret_dict

class DataProvider(object):
    def __init__(self,args, train_data):

        self.split_ratio=args.split_ratio

        self.few_shot_setting = args.few_shot_setting
        self.task_name = args.task_name
        self.fold_num = args.cv_k
        self.seed = args.seed
        self.method = args.method
        rng = random.Random(self.seed)
        self.seeds = [rng.randint(0, 10000) for _ in range(args.cv_k)]
        # self.all_data=dev32_data if self.few_shot_setting=='mdl' else train_data+dev32_data
        self.all_data = train_data
        if self.task_name == 'wsc':
            self.all_data = self.rearange_wsc_data(self.all_data)
        if self.few_shot_setting not in ['fix_setting', 'dev32_setting']:
            # assert len(train_data) == len(dev32_data)
            if self.task_name in ["multirc", 'copa']:
                cur_all_data = {}
                for item in self.all_data:
                    guid = "-".join(item.guid.split("-")[:-1])
                    if guid in cur_all_data:
                        cur_all_data[guid].append(item)
                    else:
                        cur_all_data[guid] = [item]
                cur_all_data = list(cur_all_data.items())
                self.cur_all_data = cur_all_data
                self.all_data = [y for (x, y) in self.cur_all_data]
        if self.few_shot_setting=='mdl':
            num = int(len(self.all_data)//2)
        else:
            num = int(len(self.all_data) * self.split_ratio)

        self.train_data = self.all_data[:num]
        self.dev32_data = self.all_data[num:]

        if self.task_name in ["multirc", "copa"]:
            self.train_data = list(itertools.chain.from_iterable(self.train_data))
            self.dev32_data = list(itertools.chain.from_iterable(self.dev32_data))
        if self.few_shot_setting == 'mdl':
            if self.task_name in ["multirc", 'copa']:
                self.cur_all_data = cur_all_data = self.cur_all_data[num:]
            else:
                self.all_data = self.all_data[num:]
        if self.few_shot_setting in ['cross_validation', 'mdl']:
            data_set = []
            dlen = len(self.all_data) if args.task_name not in ["multirc", 'copa'] else len(self.cur_all_data)
            num_samples_per_fold = round(dlen / self.fold_num)
            for idx in range(self.fold_num):
                start_idx = idx * num_samples_per_fold
                end_idx = dlen if idx == self.fold_num - 1 else int((idx + 1) * num_samples_per_fold)
                if args.task_name not in ["multirc", "copa"]:
                    cur_data_set = self.all_data[start_idx:end_idx]
                    data_set.append(cur_data_set)
                else:
                    cur_data_set = [data_list for (guid, data_list) in cur_all_data[start_idx:end_idx]]
                    data_set.append(list(itertools.chain.from_iterable(cur_data_set)))
            assert len(data_set) == self.fold_num
            self.data_set = data_set

        logger.info(len(self.train_data))
        logger.info(len(self.dev32_data))
        logger.info(len(self.all_data))

    def rearange_wsc_data(self, examples):
        aranged_examples = []
        for e in examples:
            if len(aranged_examples) != 0 and len(
                    set(e.text_a.split()).intersection(set(aranged_examples[-1][0].text_a.split()))) > len(
                    set(e.text_a.split())) * 0.8:
                aranged_examples[-1].append(e)
            else:
                aranged_examples.append([e])
        return aranged_examples

    def wsc_sample(self,examples,use_cloze,rng):
        new_examples=[]
        for e in examples:
            [e1,e2]=e
            if use_cloze==True:
                new_e=e1 if e1.label=='True' else e2
            else:
                rand_num = rng.random()
                new_e=e1 if rand_num < 0.5 else e2
            new_examples.append(new_e)
        return new_examples

    def get_splited_data(self, fold_id=-1):
        if self.few_shot_setting == 'fix_setting':
            if self.task_name == 'wsc':
                use_cloze=True if self.method != 'sequence_classifier' else False
                rng = random.Random(self.seed)
                return self.wsc_sample(self.train_data + self.dev32_data, use_cloze, rng), None
            return self.train_data + self.dev32_data, None
        elif self.few_shot_setting == 'dev32_setting':
            if self.task_name == 'wsc':
                use_cloze=True if self.method != 'sequence_classifier' else False
                rng = random.Random(self.seed)
                return self.wsc_sample(self.train_data,use_cloze,rng), self.wsc_sample(self.dev32_data,False,rng)
            return self.train_data, self.dev32_data
        elif self.few_shot_setting == 'dev32_split':
            fold_seed = self.seeds[fold_id]
            rng = random.Random(fold_seed)
            if self.task_name not in ['multirc', 'copa']:
                number = int(len(self.all_data) * self.split_ratio)
                all_data = copy.deepcopy(self.all_data)
                rng.shuffle(all_data)
                cur_train_data = all_data[:number]
                cur_dev32_data = all_data[number:]
            else:
                cur_all_data = copy.deepcopy(self.cur_all_data)
                rng.shuffle(cur_all_data)
                all_num = len(cur_all_data)

                number = int(all_num * self.split_ratio)
                cur_train_data_set = [data_list for (guid, data_list) in cur_all_data[:number]]
                cur_dev32_data_set = [data_list for (guid, data_list) in cur_all_data[number:]]
                cur_train_data = list(itertools.chain.from_iterable(cur_train_data_set))
                cur_dev32_data = list(itertools.chain.from_iterable(cur_dev32_data_set))
            if self.task_name == 'wsc':
                use_cloze=True if self.method != 'sequence_classifier' else False
                cur_train_data=self.wsc_sample(cur_train_data,use_cloze,rng)
                cur_dev32_data=self.wsc_sample(cur_dev32_data,False,rng)

            logger.info("train/dev data number:")
            logger.info(len(cur_train_data))
            logger.info(len(cur_dev32_data))
            return cur_train_data, cur_dev32_data
        elif self.few_shot_setting == "cross_validation":
            rng = random.Random(self.seeds[fold_id])
            cur_dev32_data = self.data_set[fold_id]
            cur_train_data = list(
                itertools.chain.from_iterable([self.data_set[j] for j in range(self.fold_num) if j != fold_id]))
            if self.task_name == 'wsc':
                use_cloze=True if self.method != 'sequence_classifier' else False
                cur_train_data=self.wsc_sample(cur_train_data,use_cloze,rng)
                cur_dev32_data=self.wsc_sample(cur_dev32_data,False,rng)
            return cur_train_data, cur_dev32_data
        elif self.few_shot_setting == "mdl":
            rng = random.Random(self.seeds[fold_id])
            cur_dev32_data = self.data_set[fold_id]
            cur_train_data = self.train_data + list(
                    itertools.chain.from_iterable([self.data_set[j] for j in range(self.fold_num) if j < fold_id]))
            if self.task_name == 'wsc':
                use_cloze=True if self.method != 'sequence_classifier' else False
                cur_train_data=self.wsc_sample(cur_train_data,use_cloze,rng)
                cur_dev32_data=self.wsc_sample(cur_dev32_data,False,rng)
            return cur_train_data, cur_dev32_data

def iterative_run(dataprovider, eval_data, wrapper_config, train_eval_config, unlabeled_data=None, aug_data=None, output_dir=None):
    output_dir = output_dir if output_dir is not None else wrapper_config.output_dir
    if train_eval_config.generations==1:
        results=run(dataprovider, eval_data, wrapper_config, train_eval_config, output_dir, unlabeled_data, aug_data, save_unlabeled_logits=False)
        return results

    for gen in range(train_eval_config.generations):
        gen_output_dir = os.path.join(output_dir, f'g{gen}')
        if gen>0:
            ipet_data_dirs={}
            if unlabeled_data is not None:
                ipet_data_dirs['unlabeled']=os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data')
            if aug_data is not None and train_eval_config.relabel_aug_data==True:
                ipet_data_dirs['aug']=os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data')
        else:
            ipet_data_dirs=None
        if wrapper_config.arch_method=='noisy_student':
            train_eval_config.use_dropout=True if gen>0 else False
        # ipet_data_dirs = os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data') if gen > 0 else None
        results=run(dataprovider, eval_data, wrapper_config, train_eval_config, gen_output_dir, unlabeled_data, aug_data, ipet_data_dirs, save_unlabeled_logits=True)

        if wrapper_config.arch_method in ['ipet', 'noisy_student']:
            assert (unlabeled_data is not None) or (aug_data is not None)
            logger.info("Augmenting data by self-labeling unlabeled data.")
            train_data, _ =dataprovider.get_splited_data(0)
            original_data_size = len(train_data) if train_data else 10 / train_eval_config.ipet_scale_factor
            num_new_examples = int(original_data_size * (train_eval_config.ipet_scale_factor ** (gen + 1)) - len(train_data))
            if unlabeled_data is not None:
                generate_ipet_train_sets(dataprovider=dataprovider,
                                        unlabeled_data=unlabeled_data,
                                        labels=wrapper_config.label_list,
                                        logits_dir=gen_output_dir,
                                        output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'),
                                        reduction="mean",
                                        num_new_examples=num_new_examples,
                                        logits_percentage=train_eval_config.ipet_logits_percentage,
                                        n_most_likely=train_eval_config.ipet_n_most_likely if gen == 0 else -1,
                                        seed=train_eval_config.seed,
                                        logits_prefix='unlabeled',
                                        use_brother_fold_logits=train_eval_config.use_brother_fold_logits)
            if aug_data is not None and train_eval_config.relabel_aug_data==True:
                generate_ipet_train_sets(dataprovider=dataprovider,
                                     unlabeled_data=aug_data,
                                     labels=wrapper_config.label_list,
                                     logits_dir=gen_output_dir,
                                     output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'),
                                     reduction="mean",
                                     num_new_examples=num_new_examples,
                                     logits_percentage=train_eval_config.ipet_logits_percentage,
                                     n_most_likely=train_eval_config.ipet_n_most_likely if gen == 0 else -1,
                                     seed=train_eval_config.seed,
                                     logits_prefix='aug',
                                     use_brother_fold_logits=train_eval_config.use_brother_fold_logits)
        elif wrapper_config.method == "flipda":
            raise NotImplementedError("FlipDA to be implemented.")
    return results


def run(dataprovider, eval_data, wrapper_config, train_eval_config, output_dir=None, unlabeled_data=None, aug_data=None, ipet_data_dirs=None, save_unlabeled_logits=False):

    pattern_ids = train_eval_config.pattern_ids
    repetitions = train_eval_config.repetitions
    folds = train_eval_config.cv_k
    seed = train_eval_config.seed
    do_train = train_eval_config.do_train
    do_eval = train_eval_config.do_eval
    if output_dir is None:
        output_dir = wrapper_config.output_dir

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))

    set_seed(seed)
    assert len(train_eval_config.sampler_seeds) >= repetitions
    for pattern_id in pattern_ids:
        for fold in range(folds):
            train_data,dev32_data=dataprovider.get_splited_data(fold_id=fold)
            for iteration in range(repetitions):
                results_dict = {}
                pattern_iter_output_dir = "{}/p{}/f{}-i{}".format(output_dir, pattern_id, fold, iteration)
                train_eval_config.sampler_seed = train_eval_config.sampler_seeds[iteration]

                if os.path.exists(pattern_iter_output_dir):
                    logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                    continue
                else:
                    os.makedirs(pattern_iter_output_dir)
                wrapper = TransformerModelWrapper(wrapper_config, pattern_id)
                if do_train:
                    ipet_train_data=None
                    if ipet_data_dirs is not None:
                        for (prefix,ipet_data_dir) in ipet_data_dirs.items():
                            p = os.path.join(ipet_data_dir, 'p{}-f{}-i{}-{}-train.bin'.format(pattern_id, fold, iteration, prefix))
                            tmp_ipet_train_data = InputExample.load_examples(p)
                            for example in tmp_ipet_train_data:
                                example.logits = None
                            if ipet_train_data is None:
                                ipet_train_data=tmp_ipet_train_data
                            else:
                                ipet_train_data+=tmp_ipet_train_data
                    if aug_data is not None and train_eval_config.relabel_aug_data==False:
                        ipet_train_data = ipet_train_data + aug_data

                    cur_results = wrapper.train(train_data, dev32_data, pattern_iter_output_dir, train_eval_config,
                                                unlabeled_data=unlabeled_data, ipet_train_data=ipet_train_data)
                    results_dict.update(cur_results)

                    with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                        fh.write(str(results_dict))

                    # if train_eval_config.few_shot_setting == "fix_setting"  or train_eval_config.every_eval_ratio==1:
                    #     logger.info("Saving trained model at {} for fix-setting.".format(pattern_iter_output_dir))
                    #     wrapper.save(pattern_iter_output_dir)

                    train_eval_config.save(os.path.join(pattern_iter_output_dir, TRAIN_EVAL_CONFIG_NAME))
                    logger.info("Saving complete")

                    if save_unlabeled_logits:
                        if unlabeled_data is not None:
                            unlabeled_logits = wrapper.evaluate(unlabeled_data,
                                train_eval_config.per_gpu_eval_batch_size,
                                train_eval_config.n_gpu,
                                train_eval_config.device,
                                train_eval_config.metrics,
                                train_eval_config.decoding_strategy,
                                train_eval_config.eval_priming,
                                train_eval_config.priming_num, priming_data=train_data)['logits']
                            save_logits(os.path.join(pattern_iter_output_dir, 'unlabeled_logits.txt'), unlabeled_logits)
                            logger.info("unlabeled logits saved.")

                        if aug_data is not None and train_eval_config.relabel_aug_data==True:
                            aug_logits = wrapper.evaluate(aug_data,
                                train_eval_config.per_gpu_eval_batch_size,
                                train_eval_config.n_gpu,
                                train_eval_config.device,
                                train_eval_config.metrics,
                                train_eval_config.decoding_strategy,
                                train_eval_config.eval_priming,
                                train_eval_config.priming_num, priming_data=train_data)['logits']
                            save_logits(os.path.join(pattern_iter_output_dir, 'aug_logits.txt'), aug_logits)
                            logger.info("augmented logits saved.")

                    wrapper.model.cpu()
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

                if do_eval:
                    logger.info("Starting evaluation...")
                    logger.info("restoring checkpoint from {}".format(pattern_iter_output_dir))
                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                    eval_result = wrapper.evaluate(
                        eval_data,
                        train_eval_config.per_gpu_eval_batch_size,
                        train_eval_config.n_gpu,
                        train_eval_config.device,
                        train_eval_config.metrics,
                        train_eval_config.decoding_strategy,
                        train_eval_config.eval_priming,
                        train_eval_config.priming_num, priming_data=train_data)

                    dev32_eval_result = wrapper.evaluate(dev32_data,
                        train_eval_config.per_gpu_eval_batch_size,
                        train_eval_config.n_gpu,
                        train_eval_config.device,
                        train_eval_config.metrics,
                        train_eval_config.decoding_strategy,
                        train_eval_config.eval_priming,
                        train_eval_config.priming_num, priming_data=train_data) if dev32_data is not None else None

                    save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
                    save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])
                    scores = eval_result['scores']
                    logger.info("--- eval_data RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                    logger.info(scores)
                    results_dict['test_set_after_training'] = scores
                    with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                        json.dump(results_dict, fh)

                    for metric, value in scores.items():
                        results[metric][pattern_id].append(value)

                    if dev32_eval_result is not None:
                        save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_eval_result)
                        save_logits(os.path.join(pattern_iter_output_dir, 'dev32_eval_logits.txt'), dev32_eval_result['logits'])
                        logger.info("--- dev32_data RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                        logger.info(dev32_eval_result["scores"])
                        results_dict["dev32_set_after_training"] = dev32_eval_result["scores"]
                        for metric, value in dev32_eval_result['scores'].items():
                            dev32_results[metric][pattern_id].append(value)

                    wrapper.model.cpu()
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        final_results=_write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
        return final_results
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def process_args(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and args.overwrite_output_dir:
        shutil.rmtree(args.output_dir)
        # pass

    if args.method == "sequence_classfier":
        args.use_cloze=False
    elif args.method in ["pet", "adapet"]:
        args.use_cloze=True
        args.use_continuous_prompt=False
    elif args.method == "ptuning":
        args.use_cloze=True
        args.use_continuous_prompt=True

    if args.arch_method=='default': #TODO
        args.generations=1

    if args.few_shot_setting in ['fix_setting','dev32_setting']:
        args.cv_k=1

    ### Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.task_name = args.task_name.lower()
    ### get metrics
    metrics = DATASETS[args.dataset_name]["metrics"]
    args.metrics = metrics.get(args.task_name, DEFAULT_METRICS)
    return args

def main():

    args = get_args()
    set_seed(args.seed)
    args = process_args(args)
    processors = DATASETS[args.dataset_name]["processors"]
    if args.task_name not in processors:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = processors[args.task_name](args.task_name)
    args.label_list = processor.get_labels()

    logger.info("\n")
    logger.info("Parameters: {}".format(args))
    logger.info("\n")

    ### prepare configurations
    data_config = get_data_config(args)
    wrapper_config = get_wrapper_config(args)
    train_eval_config = get_train_eval_config(args)

    ### prepare data
    train_data, eval_data, unlabeled_data = load_dataset(data_config)
    if args.aug_data_dir is not None:
        aug_data = processor._create_examples(args.aug_data_dir,"aug")
    else:
        aug_data = None

    logger.info('train_data: {}, eval_data: {}'.format(len(train_data),len(eval_data)))

    start_time = time.time()
    dataprovider=DataProvider(args,train_data) #,dev32_data)
    # import pdb 
    # pdb.set_trace()
    # x1,y1=dataprovider.get_splited_data(0)
    # x2,y2=dataprovider.get_splited_data(1)
    # x3,y3=dataprovider.get_splited_data(2)
    # x4,y4=dataprovider.get_splited_data(3)
    results=iterative_run(dataprovider, eval_data, wrapper_config, train_eval_config, unlabeled_data, aug_data, output_dir=args.output_dir)
    end_time=time.time()
    time_eclapsed = int(end_time-start_time)

    """
    template_name=['task_name','few_shot_setting','max_steps','warmup_ratio','gradient_accumulation_steps','per_gpu_train_batch_size','lr','pattern','max_seq_length','every_eval_ratios']
    template_values=[args.task_name,args.few_shot_setting,args.max_steps,args.warmup_step_ratio,args.gradient_accumulation_steps,args.per_gpu_train_batch_size,args.learning_rate,args.pattern_ids,args.max_seq_length,args.every_eval_ratio]
    if args.method=='ptuning':
        template_name.append('embedding_learning_rate')
        template_values.append(args.embedding_learning_rate)
    hyper_params=(': {}, '.join(template_name)+': {},').format(*template_values)
    """

    hyper_params=args.output_dir.split("/")[-1]
    result_file_name = "save_" + args.arch_method +"_"+ args.task_name \
                       + "_" + args.few_shot_setting + "_" + args.method + "_" \
                       +args.model_type + '_'+ str(len(train_data)) + ".txt"
    if args.adapet_balance_alpha!=-1:
        result_file_name = "save_detail" + args.arch_method +"_"+ args.task_name \
                    + "_" + args.few_shot_setting + "_" + args.method + "_" \
                    +args.model_type + ".txt"
    if not os.path.exists("final_results"):
        os.makedirs("final_results")
        logger.info("Creating final_results.")
    with open(os.path.join("final_results", result_file_name), "a+") as f:
        if args.task_name in ["boolq", "rte", "wic", "wsc", "copa"]:
            if 'dev32' in results and results['dev32'] is not None:
                f.write(hyper_params
                    + "\t" + str(results["dev32"]['acc'])
                    + "\t" + str(results["dev"]['acc'])
                    + "\t"  + str(time_eclapsed) +'\n')
            else:
                f.write(hyper_params
                    + "\t" + str(results["dev"]['acc'])
                    + "\t"  + str(time_eclapsed) +'\n')                
        elif args.task_name == "cb":
            if 'dev32' in results and results['dev32'] is not None:
                f.write(hyper_params
                    + "\t" + str(results["dev32"]['acc'])
                    + "\t" + str(results["dev"]['acc'])
                    + "\t" + str(results["dev32"]['f1-macro'])
                    + "\t" + str(results["dev"]['f1-macro'])
                    + "\t"  + str(time_eclapsed) + '\n')
            else:
                f.write(hyper_params
                    + "\t" + str(results["dev"]['acc'])
                    + "\t" + str(results["dev"]['f1-macro'])
                    + "\t"  + str(time_eclapsed) + '\n')
        elif args.task_name == "multirc":
            if 'dev32' in results and results['dev32'] is not None:
                f.write(hyper_params
                    + "\t" + str(results["dev32"]['acc'])
                    + "\t" + str(results["dev"]['acc'])
                    + "\t" + str(results["dev32"]['f1'])
                    + "\t" + str(results["dev"]['f1'])
                    + "\t" + str(results["dev32"]['em'])
                    + "\t" + str(results["dev"]['em'])
                    + "\t"  + str(time_eclapsed) + '\n')
            else:
                f.write(hyper_params
                    + "\t" + str(results["dev"]['acc'])
                    + "\t" + str(results["dev"]['f1'])
                    + "\t" + str(results["dev"]['em'])
                    + "\t"  + str(time_eclapsed) + '\n')                
    logger.info("\n")
    logger.info("Time elapsed: " + str(time_eclapsed))
    logger.info("\n")

if __name__ == "__main__":
    main()
