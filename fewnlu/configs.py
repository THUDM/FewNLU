import json
from abc import ABC
from typing import List


class BaseConfig(ABC):
    """Abstract class for few-shot configuration that can be saved to and loaded from json files."""
    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save the configuration to a json file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a configuration from a json file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class DatasetConfig(BaseConfig):
    def __init__(self, use_cloze:bool, dataset_name:str, task_name:str, data_dir:str, seed:int, train_examples:int,
                 dev_examples:int, dev32_examples:int, unlabeled_examples:int, test_examples:int, eval_set:str, method:str,
                 split_examples_evenly:bool, label_list:List, **kwargs): #TODO add kwargs
        self.use_cloze = use_cloze
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.seed = seed
        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.dev32_examples = dev32_examples
        self.unlabeled_examples = unlabeled_examples #TODO
        self.test_examples = test_examples
        self.eval_set = eval_set
        self.method = method
        self.split_examples_evenly = split_examples_evenly
        self.label_list = label_list
        self.task_name= task_name

def get_data_config(args):
    return DatasetConfig(**vars(args)) #TODO
    # return DatasetConfig(use_cloze=args.use_cloze,
    #                      dataset_name=args.dataset_name,
    #                      data_dir=args.data_dir,
    #                      seed=args.seed,
    #                      train_examples=args.train_examples,
    #                      dev_examples=args.dev_examples,
    #                      dev32_examples=args.dev32_examples,
    #                      test_examples=args.test_examples,
    #                      eval_set=args.eval_set,
    #                      method=args.method,
    #                      split_examples_evenly=args.split_examples_evenly,
    #                      task_name=args.task_name,
    #                      label_list=args.label_list)



class WrapperConfig(BaseConfig):
    """A configuration for a :class:`TransformerModelWrapper`."""
    def __init__(self,
                 model_type: str,
                 model_name_or_path: str,
                 dataset_name: str,
                 task_name: str,
                 max_seq_length: int,
                 label_list: List[str],
                 verbalizer_file: str,
                 cache_dir: str,
                 method,
                 arch_method,
                 use_continuous_prompt: str,
                 prompt_encoder_head_type: str,
                 output_dir: str,
                 fix_deberta: bool,
                 use_cloze: float,
                 dropout_rate: float,
                 seed: int,
                 device=None, **kwargs):

        self.seed = seed
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir

        self.dataset_name=dataset_name
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.verbalizer_file = verbalizer_file

        self.method = method
        self.arch_method = arch_method #TODO
        self.use_continuous_prompt = use_continuous_prompt
        self.prompt_encoder_head_type = prompt_encoder_head_type
        self.fix_deberta = fix_deberta
        self.use_cloze = use_cloze

        self.dropout_rate = dropout_rate

        self.output_dir = output_dir
        self.device = device

        if self.method=='adapet':
            self.adapet_mask_alpha=kwargs['adapet_mask_alpha']
            self.max_num_lbl_tok=kwargs['max_num_lbl_tok']
            self.adapet_balance_alpha=kwargs['adapet_balance_alpha']



class TrainEvalConfig(BaseConfig):
    """Configuration for training/evaluating a model."""
    def __init__(self,
                 per_gpu_train_batch_size:int,
                 per_gpu_unlabeled_batch_size:int,
                 per_gpu_eval_batch_size:int,
                 gradient_accumulation_steps:int,
                 num_train_epochs:int,
                 max_steps:int,
                 learning_rate: float,
                 embedding_learning_rate: float,
                 max_grad_norm: float,
                 weight_decay:float,
                 adam_epsilon:float,
                 warmup_step_ratio: float,
                 alpha:float,
                 temperature:float,
                #  every_eval_step:int,
                 every_eval_ratio:float,
                 do_train:bool,
                 do_eval:bool,
                 repetitions:int,
                 cv_k:int,
                 seed:int,
                 sampler_seeds:List[int],
                 device:str,
                 n_gpu:int,
                 metrics: List[str],
                 pattern_ids:List[int],
                 train_priming:bool,
                 eval_priming:bool,
                 priming_num:int,
                 early_stop_epoch:int,
                 reduction:str,
                 decoding_strategy:str,
                 generations:int,
                 ipet_scale_factor:float,
                 ipet_n_most_likely:float,
                 ipet_logits_percentage:float,
                 use_brother_fold_logits:bool,
                 use_dropout:bool,
                 lm_training:bool,
                #  dropout_rate:float,
                 few_shot_setting:str,
                 relabel_aug_data:bool, **kwargs):

        self.few_shot_setting = few_shot_setting
        # self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.ipet_logits_percentage = ipet_logits_percentage
        self.ipet_n_most_likely = ipet_n_most_likely
        self.ipet_scale_factor = ipet_scale_factor
        self.use_brother_fold_logits = use_brother_fold_logits
        self.generations = generations

        self.device = device
        self.n_gpu = n_gpu
        self.pattern_ids=pattern_ids
        self.metrics = metrics

        self.train_priming = train_priming
        self.eval_priming = eval_priming
        self.priming_num = priming_num

        self.sampler_seeds = sampler_seeds
        self.seed = seed

        self.repetitions = repetitions
        self.cv_k = cv_k
        self.do_eval = do_eval
        self.do_train = do_train
        # self.every_eval_step = every_eval_step
        self.every_eval_ratio = every_eval_ratio

        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_unlabeled_batch_size = per_gpu_unlabeled_batch_size
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps

        self.alpha = alpha
        self.temperature = temperature
        self.warmup_step_ratio = warmup_step_ratio
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.embedding_learning_rate = embedding_learning_rate

        self.early_stop_epoch = early_stop_epoch

        self.reduction=reduction
        self.decoding_strategy=decoding_strategy

        self.relabel_aug_data=relabel_aug_data
        self.lm_training=lm_training


"""
class IPetConfig(PetConfig):
    # Configuration for iterative PET training.

    def __init__(self, generations: int = 3, logits_percentage: float = 0.25, scale_factor: float = 5,
                 n_most_likely: int = -1):
        self.generations = generations
        self.logits_percentage = logits_percentage
        self.scale_factor = scale_factor
        self.n_most_likely = n_most_likely
"""



def get_wrapper_config(args):
    return WrapperConfig(**vars(args)) #TODO
    """
    model_cfg = WrapperConfig(model_type=args.model_type,
                 model_name_or_path=args.model_name_or_path,
                 dataset_name=args.dataset_name,
                 task_name=args.task_name,
                 max_seq_length=args.max_seq_length,
                 label_list=args.label_list,
                 verbalizer_file=args.verbalizer_file,
                 cache_dir=args.cache_dir,
                 method=args.method,
                 use_continuous_prompt=args.use_continuous_prompt,
                 prompt_encoder_head_type=args.prompt_encoder_head_type,
                 output_dir=args.output_dir,
                 fix_deberta=args.fix_deberta,
                 use_cloze=args.use_cloze,
                 device=args.device, seed=args.seed)

    return model_cfg
    """


def get_train_eval_config(args):
    return TrainEvalConfig(**vars(args)) #TODO
    """
    train_cfg = TrainEvalConfig(per_gpu_train_batch_size=args.per_gpu_train_batch_size,
                 per_gpu_unlabeled_batch_size=args.per_gpu_unlabeled_batch_size,
                 per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
                 gradient_accumulation_steps=args.gradient_accumulation_steps,
                 num_train_epochs=args.num_train_epochs,
                 max_steps=args.max_steps,
                 learning_rate=args.learning_rate,
                 embedding_learning_rate=args.embedding_learning_rate,
                 max_grad_norm=args.max_grad_norm,
                 weight_decay=args.weight_decay,
                 adam_epsilon=args.adam_epsilon,
                 warmup_step_ratio=args.warmup_step_ratio,
                 alpha=args.alpha,
                 metrics=args.metrics,
                 temperature=args.temperature,
                 every_eval_step=args.every_eval_step,
                 do_train=args.do_train,
                 do_eval=args.do_eval,
                 repetitions=args.repetitions,
                 seed=args.seed,
                 sampler_seed=args.sampler_seed,
                 device=args.device,
                 n_gpu=args.n_gpu,
                 pattern_ids=args.pattern_ids,
                 train_priming=args.train_priming,
                 eval_priming=args.eval_priming,
                 priming_num=args.priming_num,
                 reduction=args.reduction,
                 decoding_strategy=args.decoding_strategy,
                early_stop_epoch=args.early_stop_epoch,
                ipet_generations=args.ipet_generations,
                ipet_scale_factor=args.ipet_scale_factor,
                ipet_n_most_likely=args.ipet_n_most_likely,
                ipet_logits_percentage=args.ipet_logits_percentage,
                dropout_rate=args.dropout_rate)
    return train_cfg
    """