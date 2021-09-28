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
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""

import jsonpickle
import os
from typing import List, Dict
import random

import torch
import numpy as np
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import Sampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup

# TODO: for deberta-v2
from configs import WrapperConfig
from transformers import __version__ as transformers_version
from transformers.data.metrics import simple_accuracy

import log
from data_utils.preprocessor import PREPROCESSORS
from methods.method_vars import METHOD_CLASSES
from data_utils.task_helpers import TASK_HELPERS

from global_vars import *
from utils import InputFeatures, DictDataset, exact_match, get_verbalization_ids


logger = log.get_logger()

# todo
# Re-implement RandomSampler to decouple independent random seed.
# class RandomSampler(Sampler):
#     def __init__(self, data_source, seed) -> None:
#         super(RandomSampler, self).__init__(data_source)
#         self.data_source = data_source
#         self.seed = seed

#     @property
#     def num_samples(self) -> int:
#         return len(self.data_source)

#     def __iter__(self):
#         n = len(self.data_source)
#         generator = torch.Generator()
#         generator.manual_seed(self.seed)
#         yield from torch.randperm(n, generator=generator).tolist()

#     def __len__(self):
#         return len(self.data_source)

class RandomSampler(Sampler):
    def __init__(self, data_source, seed) -> None:
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.seed = seed
        self.rng = random.Random(self.seed)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        generator = torch.Generator()
        generator.manual_seed(self.rng.randint(0,10000))
        yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self):
        return len(self.data_source)


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""
    def __init__(self, config: WrapperConfig, pattern_id: int):
        self.config = config
        self.config.pattern_id = pattern_id
        self.config.wrapper_type = "cls" if self.config.method == "sequence_classifier" else "mlm"
        logger.info("WrapperConfig: ")
        logger.info(self.config)

        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            self.config.model_name_or_path, cache_dir=self.config.cache_dir if self.config.cache_dir else None)
        """
        if self.config.model_type == 'gpt2':
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token
        """
        logger.info("Tokenizer Loaded.")

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](
            self.tokenizer, self.config.dataset_name, self.config.task_name, self.config.pattern_id,
            self.config.use_cloze, self.config.use_continuous_prompt, self.config.max_seq_length,
            self.config.label_list, self.config.seed)

        if config.method not in METHOD_CLASSES:
            raise NotImplementedError(f"Training method '{config.method}' not implemented.")
        self.model = METHOD_CLASSES[config.method](self.config, self.tokenizer, self.preprocessor.pvp)

        self.task_helper = TASK_HELPERS[self.config.task_name](
            self.config, self.tokenizer, self.preprocessor, self.model) if self.config.task_name in TASK_HELPERS else \
            None

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](wrapper.tokenizer,
                wrapper.config.dataset_name, wrapper.config.task_name, wrapper.config.pattern_id,
                wrapper.config.use_cloze, wrapper.config.use_continuous_prompt, wrapper.config.max_seq_length,
                wrapper.config.label_list, wrapper.config.seed)
        wrapper.model = METHOD_CLASSES[wrapper.config.method](wrapper.config, wrapper.tokenizer, wrapper.preprocessor.pvp)
        wrapper.model.model = model_class.from_pretrained(path)
        if wrapper.config.method == "ptuning":
            save_path_file = os.path.join(path, "embeddings.pth")
            data = torch.load(save_path_file)
            wrapper.model.prompt_encoder.load_state_dict(data["prompt_encoder"])
        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](
            wrapper.config, wrapper.tokenizer, wrapper.preprocessor, wrapper.model) if wrapper.config.task_name in \
                                                                                 TASK_HELPERS else None
        if torch.cuda.device_count() > 1:
            wrapper.model = torch.nn.DataParallel(wrapper.model)
        wrapper.model.cuda()
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)
        if self.config.method == "ptuning":
            state = {"prompt_encoder": model_to_save.prompt_encoder.state_dict()}
            save_path_file = os.path.join(path, "embeddings.pth")
            torch.save(state, save_path_file)
        return

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self, train_data, dev32_data, pattern_iter_output_dir, train_eval_config, unlabeled_data=None, ipet_train_data=None):
        #TODO jing
        if not ipet_train_data:
            ipet_train_data = []
        train_data=train_data+ipet_train_data
        # if self.config.method in SELF_TRAINING_METHODS:
        #     assert aug_train_data is not None
        #     train_data = train_data + aug_train_data
        if self.config.method in UNLABELED_BASED_METHODS:
            assert unlabeled_data is not None
        results_dict = {}
        """
        global_step, tr_loss = self._train(train_data=train_data,
                                           dev32_data=dev32_data,
                                           unlabeled_data=unlabeled_data,
                                           per_gpu_train_batch_size=train_eval_config.per_gpu_train_batch_size,
                                           per_gpu_eval_batch_size=train_eval_config.per_gpu_eval_batch_size,
                                           per_gpu_unlabeled_batch_size=train_eval_config.per_gpu_unlabeled_batch_size,
                                           learning_rate=train_eval_config.learning_rate,
                                           embedding_learning_rate=train_eval_config.embedding_learning_rate,
                                           weight_decay=train_eval_config.weight_decay,
                                           adam_epsilon=train_eval_config.adam_epsilon,
                                           warmup_step_ratio=train_eval_config.warmup_step_ratio,
                                           gradient_accumulation_steps=train_eval_config.gradient_accumulation_steps,
                                           max_grad_norm=train_eval_config.max_grad_norm,
                                           train_priming=train_eval_config.train_priming,
                                           sampler_seed=train_eval_config.sampler_seed,
                                           eval_priming=train_eval_config.eval_priming,
                                           priming_num=train_eval_config.priming_num,
                                           max_steps=train_eval_config.max_steps,
                                           num_train_epochs=train_eval_config.num_train_epochs,
                                           metrics=train_eval_config.metrics,
                                           decoding_strategy=train_eval_config.decoding_strategy,
                                           alpha=train_eval_config.alpha,
                                           temperature=train_eval_config.temperature,
                                           early_stop_epoch=train_eval_config.early_stop_epoch,
                                           every_eval_step=train_eval_config.every_eval_step,
                                           pattern_iter_output_dir=pattern_iter_output_dir,
                                           n_gpu=train_eval_config.n_gpu, device=train_eval_config.device)
        """
        global_step, tr_loss = self._train(train_data=train_data,
                                           dev32_data=dev32_data,
                                           unlabeled_data=unlabeled_data,
                                           pattern_iter_output_dir=pattern_iter_output_dir,
                                           ** train_eval_config.__dict__)
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss
        return results_dict


    def evaluate(self, eval_data, per_gpu_eval_batch_size, n_gpu, device, metrics, decoding_strategy, eval_priming,
                 priming_num, priming_data=None, use_dropout=False):
        if eval_priming and priming_data:
            for example in eval_data:
                # priming_example = random.sample(priming_data, k=priming_num)
                priming_example = priming_data
                example.meta['priming_data'] = priming_example
        results = self._eval(eval_data,
                             device,
                             per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                             n_gpu=n_gpu,
                             decoding_strategy=decoding_strategy,
                             eval_priming=eval_priming,
                             priming_num=priming_num,
                             use_dropout=use_dropout)
        predictions = np.argmax(results['logits'], axis=1)

        scores = {}
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            else:
                raise ValueError(f"Metric '{metric}' not implemented")

        results['scores'] = scores
        results['predictions'] = predictions
        return results

    def _prepare_dataloader(self, type, data, per_gpu_batch_size, use_priming, sampler_seed, n_gpu, labelled):
        batch_size = per_gpu_batch_size * max(1, n_gpu)
        dataset = self._generate_dataset(data, priming=use_priming, labelled=labelled)

        if type == "train" or type == "extra":
            sampler = RandomSampler(dataset, sampler_seed)
        elif type == 'dev32' or type == "eval":
            sampler = SequentialSampler(dataset)
        else:
            raise NotImplementedError("Type {} not implemented".format(type))
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader


    def _prepare_optimizer_and_scheduler(self, t_total, warmup_step_ratio, weight_decay, learning_rate,
                                         embedding_learning_rate, adam_epsilon):
        cur_model = self.model.module if hasattr(self.model, 'module') else self.model
        warmup_steps = int(t_total * warmup_step_ratio)
        no_decay = ['bias', 'LayerNorm.weight']
        print(cur_model.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        ret_dict = {"optimizer": optimizer, "scheduler": scheduler}
        if self.config.method == "ptuning":
            embedding_parameters = [{'params': [p for p in cur_model.prompt_encoder.parameters()]}]
            embedding_optimizer = AdamW(embedding_parameters, lr=embedding_learning_rate, eps=adam_epsilon)
            embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

            ret_dict["embedding_optimizer"] = embedding_optimizer
            ret_dict["embedding_scheduler"] = embedding_scheduler
        return ret_dict

    def _train(self, train_data, dev32_data, unlabeled_data,
               per_gpu_train_batch_size, per_gpu_eval_batch_size, per_gpu_unlabeled_batch_size,
               learning_rate,embedding_learning_rate,weight_decay,adam_epsilon,warmup_step_ratio,
               gradient_accumulation_steps,max_grad_norm,
               train_priming, sampler_seed, eval_priming, priming_num,
               max_steps, num_train_epochs,
               metrics, decoding_strategy, alpha, temperature,
               early_stop_epoch, every_eval_ratio, pattern_iter_output_dir,
               n_gpu, device, few_shot_setting, use_dropout=False, lm_training=False, **_):
        # import pdb
        # pdb.set_trace()
        train_dataloader=self._prepare_dataloader(
            "train", train_data, per_gpu_train_batch_size, train_priming, sampler_seed, n_gpu, labelled=True)
        if self.config.method == "adapet":
            extra_dataloader = self._prepare_dataloader(
                "extra", train_data, per_gpu_train_batch_size, train_priming, sampler_seed, n_gpu, labelled=True)
            extra_iter = extra_dataloader.__iter__()
        else:
            extra_dataloader, extra_iter = None, None

        if lm_training==True:
            lm_dataloader = self._prepare_dataloader(
                "extra", unlabeled_data, per_gpu_unlabeled_batch_size, train_priming, sampler_seed, n_gpu,labelled=False)
            lm_iter = lm_dataloader.__iter__()
        else:
            lm_dataloader, lm_iter = None, None

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        every_eval_step = int(every_eval_ratio * t_total)
        logger.info("\n")
        logger.info("num_steps_per_dataset:")
        logger.info(len(train_dataloader) // gradient_accumulation_steps)
        logger.info("total_steps:")
        logger.info(t_total)
        logger.info("num_train_epochs:")
        logger.info(num_train_epochs)

        ret_dict = self._prepare_optimizer_and_scheduler(
            t_total, warmup_step_ratio, weight_decay, learning_rate, embedding_learning_rate, adam_epsilon)
        optimizer, scheduler = ret_dict["optimizer"], ret_dict["scheduler"]
        embedding_optimizer, embedding_scheduler = None, None
        if self.config.method == "ptuning":
            embedding_optimizer, embedding_scheduler = ret_dict["embedding_optimizer"], ret_dict["embedding_scheduler"]

        writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "writer_logs"))

        logger.info("\n")
        logger.info("dev32_data performance before training.")
        dev32_scores = self.evaluate(
            dev32_data, per_gpu_eval_batch_size, n_gpu, device, metrics, decoding_strategy, eval_priming,
            priming_num, priming_data=train_data,use_dropout=False)["scores"] if dev32_data is not None else None
        logger.info(dev32_scores)
        logger.info("\n")


        best_scores = 0.0
        cur_early_stop_epoch = 0
        best_global_step, best_tr_loss = 0, 0.0

        if few_shot_setting == "fix_setting" or every_eval_ratio==1:
            early_stop_epoch = num_train_epochs + 1

        step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for _, batch in enumerate(epoch_iterator):
                # import pdb 
                # pdb.set_trace()
                self.model.train()
                batch = {k: t.to(device) for k, t in batch.items()}
                extra_batch = None
                lm_batch = None

                if extra_dataloader is not None:
                    while extra_batch is None:
                        try:
                            extra_batch = extra_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting adapet dataset")
                            extra_iter = extra_dataloader.__iter__()

                if lm_dataloader is not None:
                    while lm_batch is None:
                        try:
                            lm_batch = lm_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting lm dataset")
                            lm_iter = lm_dataloader.__iter__()

                if self.config.method == "adapet":
                    extra_batch=self.process_adapet_batch(extra_batch)
                    extra_batch={k:t.to(device) for k,t in extra_batch.items()}

                train_step_inputs = {
                    "extra_batch": extra_batch,
                    "alpha": alpha,
                    "temperature": temperature,
                    "priming": train_priming,
                    "priming_num": priming_num,
                    "use_dropout": use_dropout,
                }
                if self.config.method=='adapet':
                    loss = self.model.train_step(batch, **train_step_inputs)
                else:
                    loss = self.task_helper.train_step(batch, **train_step_inputs) if (self.task_helper and (self.config.wrapper_type == "mlm")) else None
                    if loss is None:
                        loss = self.model.train_step(batch, **train_step_inputs)

                if lm_training==True:
                    # unlabeled_lm_input_ids = lm_batch['input_ids']
                    # unlabeled_lm_block_flags = lm_batch["block_flags"]
                    # lm_batch['input_ids'], lm_batch['mlm_labels'] = self._mask_tokens(unlabeled_lm_input_ids)
                    lm_batch['input_ids'], lm_batch['mlm_labels'] = self._mask_tokens(lm_batch)
                    lm_batch = {k: t.to(device) for k, t in lm_batch.items()}
                    lm_inputs=self.model.generate_default_inputs(lm_batch)
                    lm_inputs['labels']=lm_batch['mlm_labels']
                    lm_loss=self.model(**lm_inputs)[0]
                    # lm_loss.backward()
                    # tr_loss+=lm_loss.item()
                else:
                    lm_loss=None
                    
                loss = loss + lm_loss if lm_loss != None else loss
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:

                    loss_scalar = (tr_loss - logging_loss) / gradient_accumulation_steps
                    writer.add_scalar("train_loss", loss_scalar, global_step=global_step)
                    logging_loss = tr_loss

                    learning_rate_scalar = scheduler.get_lr()[0]
                    writer.add_scalar("learning_rate", learning_rate_scalar, global_step=global_step)

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    if self.config.method == "ptuning":
                        embedding_optimizer.step()
                        embedding_scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if every_eval_step > 0 and global_step % every_eval_step == 0 and few_shot_setting != "fix_setting" and every_eval_ratio!=1:
                        dev32_scores_dict = self.evaluate(dev32_data, per_gpu_eval_batch_size, n_gpu, device, metrics,
                                                     decoding_strategy, eval_priming, priming_num,
                                                     priming_data=train_data,use_dropout=use_dropout)["scores"]
                        for metric in metrics:
                            writer.add_scalar("dev32_" + metric, dev32_scores_dict[metric], global_step=global_step)

                        # if few_shot_setting != "dev32_setting":
                        #     logger.info("cur_dev32_scores:")
                        #     logger.info(dev32_scores_dict)
                        #     continue

                        save_checkpoint = True
                        cur_dev32_score = np.mean(list(dev32_scores_dict.values()))
                        if cur_dev32_score < best_scores:
                            save_checkpoint = False

                        if save_checkpoint:
                            if best_scores == cur_dev32_score:
                                cur_early_stop_epoch += 1
                            else:
                                cur_early_stop_epoch = 0

                            best_tr_loss = tr_loss
                            best_scores = cur_dev32_score
                            best_global_step = global_step

                            logger.info("\n")
                            logger.info("best_global_step: {} | saving models at {}...".format(best_global_step, pattern_iter_output_dir))
                            logger.info("best_scores:")
                            logger.info(dev32_scores_dict)
                            self.save(pattern_iter_output_dir)
                        else:
                            cur_early_stop_epoch += 1
                        logger.info("early_stop_epoch: " + str(cur_early_stop_epoch))
                        logger.info("\n\n")

                if 0 < max_steps < global_step or cur_early_stop_epoch >= early_stop_epoch:
                    epoch_iterator.close()
                    break
                step += 1
            if 0 < max_steps < global_step or cur_early_stop_epoch >= early_stop_epoch:
                train_iterator.close()
                break

        if few_shot_setting == "fix_setting"  or every_eval_ratio==1:
            return global_step, (tr_loss / global_step if global_step > 0 else -1)
        else:
            return best_global_step, (best_tr_loss / best_global_step if best_global_step > 0 else -1)


    def _eval(self, eval_data, device, per_gpu_eval_batch_size, n_gpu, eval_priming, priming_num, decoding_strategy, use_dropout=False) -> Dict:
        eval_dataloader = self._prepare_dataloader(
            "eval", eval_data, per_gpu_eval_batch_size, eval_priming, None, n_gpu, labelled=True)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.to(device) for k, t in batch.items()}
            eval_step_inputs={"use_dropout": use_dropout}
            labels = batch['labels']
            indices = batch['idx']
            # import pdb 
            # pdb.set_trace()
            with torch.no_grad():
                if self.config.method=='adapet':
                    logits = self.model.eval_step(batch,**eval_step_inputs)
                else:
                    # some tasks require special evaluation
                    logits = self.task_helper.eval_step(batch, decoding_strategy=decoding_strategy,**eval_step_inputs) if (self.task_helper
                                                                                                        and (self.config.wrapper_type == "mlm")) else None
                    if logits is None:
                        logits = self.model.eval_step(batch,**eval_step_inputs)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        return {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }



    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False):
        features = self._convert_examples_to_features(data, labelled=labelled, priming=priming)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            "block_flags": torch.tensor([f.block_flags for f in features], dtype=torch.long)
        }
        """
        if self.wrapper_type == PLM_WRAPPER:
            feature_dict['perm_mask'] = torch.tensor([f.perm_mask for f in features], dtype=torch.float)
            feature_dict['target_mapping'] = torch.tensor([f.target_mapping for f in features], dtype=torch.float)
        """


        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      priming: bool = False) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming=priming)

            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)

            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features

    # def _mask_tokens(self, original_input_ids):

    #     """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    #     input_ids = original_input_ids.clone()

    #     labels = input_ids.clone()

    #     # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    #     probability_matrix = torch.full(labels.shape, 0.15)
    #     special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
    #                            labels.tolist()]
    #     probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    #     masked_indices = torch.bernoulli(probability_matrix).bool()

    #     # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
    #     if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
    #         ignore_value = -100
    #     else:
    #         ignore_value = -1

    #     labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

    #     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    #     input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    #     # 10% of the time, we replace masked input tokens with random word
    #     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #     random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    #     input_ids[indices_random] = random_words[indices_random]

    #     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #     return input_ids, labels


    def _mask_tokens(self, batch):

        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        input_ids = batch['input_ids'].clone()
        labels=batch['labels'].clone()
        # if self.config.task_name not in ['wsc','copa']:
        #     for idx in range(len(input_ids)):
        #         lab_map = {i:lab for i, lab in enumerate(self.config.label_list)}
        #         lab = lab_map[labels[idx].item()]
        #         label_token = self.preprocessor.pvp.verbalize(lab)
        #         label_id = get_verbalization_ids(label_token[0], self.tokenizer, True)
        #         input_ids[idx] = torch.where(input_ids[idx]==self.tokenizer.mask_token_id, torch.tensor(label_id), input_ids[idx])
        # elif self.config.task_name == "copa":
        #     for idx in range(len(labels)):
        #         if labels[idx].item() == 0:
        #             lbl_choice = batch["choice1_token_ids"][idx]
        #             input_ids[idx] = torch.where(batch['input_ids1'][idx] == self.tokenizer.mask_token_id, lbl_choice, batch['input_ids1'][idx])
        #         elif labels[idx].item() == 1:
        #             lbl_choice = batch["choice2_token_ids"][idx]
        #             input_ids[idx] = torch.where(batch['input_ids2'][idx] == self.tokenizer.mask_token_id, lbl_choice, batch['input_ids2'][idx])
        #         else:
        #             raise ValueError("Invalid Lbl")
        # elif self.config.task_name == "wsc":
        #     for idx in range(len(input_ids)):
        #         target_ids = batch["target_token_ids"][idx]
        #         input_ids[idx] = torch.where(input_ids[idx] == self.tokenizer.mask_token_id, target_ids, input_ids[idx])

        labels = input_ids.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        masked_idxs=(input_ids==self.tokenizer.mask_token_id).bool()
        # for idx in range(len(labels)):
        #     labels[idx]=torch.where(input_ids[idx]==self.tokenizer.mask_token_id,ignore_value,input_ids[idx])
        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens
        labels[masked_idxs]=ignore_value
        

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def process_adapet_batch(self,batch):
        input_ids = batch['input_ids'].clone()
        if self.config.task_name not in ["wsc", "copa"]:
            labels=batch['labels']
            fake_labels=np.random.randint(len(self.config.label_list),size=len(input_ids))
            tgt=torch.from_numpy(fake_labels).long()==labels
            for idx in range(len(input_ids)):
                lab_map = {i:lab for i, lab in enumerate(self.config.label_list)}
                lab = lab_map[fake_labels[idx]]
                fake_label_token = self.preprocessor.pvp.verbalize(lab)
                fake_label_id = get_verbalization_ids(fake_label_token[0], self.tokenizer, True)
                input_ids[idx] = torch.where(input_ids[idx]==self.tokenizer.mask_token_id, torch.tensor(fake_label_id), input_ids[idx])
        elif self.config.task_name == "copa":
            labels=batch['labels']
            fake_labels=np.random.randint(len(self.config.label_list),size=len(input_ids))
            tgt=torch.from_numpy(fake_labels).long()==labels
            for idx in range(len(labels)):
                if batch["labels"][idx].item() == 0:
                    lbl_choice = batch["choice1_token_ids"][idx]
                    input_ids[idx] = torch.where(batch['input_ids1'][idx] == self.tokenizer.mask_token_id, lbl_choice, batch['input_ids1'][idx])
                elif batch["labels"][idx].item() == 1:
                    lbl_choice = batch["choice2_token_ids"][idx]
                    input_ids[idx] = torch.where(batch['input_ids2'][idx] == self.tokenizer.mask_token_id, lbl_choice, batch['input_ids2'][idx])
                else:
                    raise ValueError("Invalid Lbl")
        elif self.config.task_name == "wsc":
            tgt = torch.ones(input_ids.shape[0]).long()
            for idx in range(len(input_ids)):
                target_ids = batch["target_token_ids"][idx]
                input_ids[idx] = torch.where(input_ids[idx] == self.tokenizer.mask_token_id, target_ids, input_ids[idx])
        else:
            raise ValueError("invalid.")

        mask_input_ids = input_ids.clone()
        # sample_length = min(tot_length, self.config.max_seq_length)
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        sample_lengths=(input_ids!=0).sum(dim=1).tolist()
        for (sample_id,sample_length) in enumerate(sample_lengths):
            upto_ratio_mask = np.random.rand()
            num_sample = max(int(upto_ratio_mask * self.config.adapet_mask_alpha * sample_length), 2) - 1
            mask_idx = random.sample(range(0, sample_length), k=num_sample)
            mask_idx = np.asarray(mask_idx)
            not_masked_idx=np.asarray([i for i in range(self.config.max_seq_length) if i not in mask_idx])
            mask_input_ids[sample_id,mask_idx]=self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            # labels[sample_id,not_masked_idx] = ignore_value
        # import pdb 
        # pdb.set_trace()
        batch["input_ids"] = input_ids
        batch["tgt"]=tgt
        batch["mask_input_ids"]=mask_input_ids
        # batch["mlm_labels"]=labels
        return batch