import argparse

from global_vars import MODEL_CLASSES
from methods.method_vars import METHOD_CLASSES,ARCH_METHOD_CLASSES


def add_few_shot_setting_args(parser):
    group = parser.add_argument_group("few-shot setting")
    # group.add_argument('--every_eval_step', type=int, default=5, help="Eval dev32 every X updates steps.")
    group.add_argument('--every_eval_ratio',type=float,default=0.02)
    group.add_argument('--early_stop_epoch', type=int, default=6, help="Maximum early stop epoch.")
    group.add_argument('--few_shot_setting', type=str, default='dev32_setting', choices=['fix_setting', "dev32_setting", "cross_validation", "mdl", "dev32_split"],
                       help="Which few-shot setting to use.")
    group.add_argument('--cv_k',type=int,default=4)
    return parser


def add_required_args(parser):
    parser.add_argument("--method", type=str, required=True, choices=METHOD_CLASSES.keys(),
                        help="The training method to use. Either regular sequence classification, PET, pTuning, or AdaPET")
    parser.add_argument("--arch_method", type=str, required=True, choices=ARCH_METHOD_CLASSES,
                        help="The training architecture to use. Either ipet or noisy_student")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the data_utils.")
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--dataset_name", default='superglue', required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the data_utils to "
                                                                                   "train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    return parser


def add_training_args(parser):
    parser.add_argument("--repetitions", default=1, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")
    parser.add_argument("--cache_dir", default="", type=str, help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_step_ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--sampler_seeds', type=int, default=[10, 20, 30], nargs='+', help='Control train sample order.')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--use_cloze", action='store_true')

    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling data_utils (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")

    parser.add_argument("--verbalizer_file", default=None, help="The path to a file to override default verbalizers (only for PET)")
    parser.add_argument("--reduction", default='mean', choices=['wmean', 'mean'],
                        help="Reduction strategy for merging predictions from multiple PET models. Select either "
                             "uniform weighting (mean) or weighting based on train set accuracy (wmean)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")

    return parser


def add_data_args(parser):
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--dev32_examples", default=-1, type=int,
                        help="The total number of dev32 examples to use, where -1 equals all examples.")
    parser.add_argument("--dev_examples", default=-1, type=int,
                        help="The total number of dev examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--aug_data_dir",default=None,type=str, help="The augmented data address")
    parser.add_argument("--relabel_aug_data",action="store_true",
                        help="Whether the labels of augmented data should be tagged by the model")

    return parser



def add_adapet_args(parser): #TODO
    parser.add_argument("--adapet_mask_alpha",default=0.105,type=float, help="mask alpha")
    parser.add_argument("--adapet_balance_alpha",default=-1,type=float, help="balance alpha")
    parser.add_argument("--max_num_lbl_tok",default=20,type=int)
    return parser 

def get_args():
    parser = argparse.ArgumentParser(description="Command line interface for fewnlu.")
    # Required parameters
    parser = add_required_args(parser)
    # Training parameters
    parser = add_training_args(parser)
    # Data-related parameters
    parser = add_data_args(parser)
    # Few-shot setting parameters
    parser = add_few_shot_setting_args(parser)
    # Adapet setting parameters
    parser = add_adapet_args(parser) #TODO

    # others: for priming
    parser.add_argument('--eval_priming', action='store_true', help="Whether to use priming for evaluation")
    parser.add_argument('--train_priming', action='store_true', help="Whether to use priming for evaluation")
    parser.add_argument('--priming_num', type=int, help="")

    # others: for continuous prompt
    parser.add_argument("--embedding_learning_rate", default=1e-4, type=float)
    parser.add_argument("--use_continuous_prompt", action='store_true', help="If set to true, use P-tuning.")
    parser.add_argument("--prompt_encoder_head_type", type=str, default="raw")

    # others: for ipet
    parser.add_argument("--generations", default=3, type=int, help="The number of generations to train (only for iPET)")
    parser.add_argument("--ipet_logits_percentage", default=0.25, type=float, help="The percentage of models to "
                                                                                   "choose for annotating new training sets (only for iPET)")
    parser.add_argument("--ipet_scale_factor", default=5, type=float, help="The factor by which to increase the "
                                                                           "training set size per generation (only for iPET)")
    parser.add_argument("--ipet_n_most_likely", default=-1, type=int, help="If >0, in the first generation the "
                                                                           "n_most_likely examples per label are "
                                                                           "chosen even if their predicted label is "
                                                                           "different (only for iPET)")
    parser.add_argument("--use_brother_fold_logits",action="store_true")

    # others: for deberta
    parser.add_argument("--fix_deberta", action='store_true', help="If set to true, fix 1/3 parameters.")

    # others: noisy student dropout
    parser.add_argument("--use_dropout",action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.05)

    # parser.add_argument("--do_multi_verbalizer", action="store_true", help="Whether to use multi-verbalizer")
    parser.add_argument("--lm_training", action='store_true', help="Whether to use language modeling as auxiliary data_utils (only for PET)")
    """
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    # todo: distillation or not
    parser.add_argument("--no_distillation", action='store_true', help="If set to true, no distillation is performed (only for PET)")
    # todo: lm_task
    parser.add_argument("--lm_training", action='store_true', help="Whether to use language modeling as auxiliary data_utils (only for PET)")
    parser.add_argument("--use_cali", action='store_true',
                        help="whether to use calibration.")
    parser.add_argument("--use_adapet_loss", action='store_true', help="whether to use adapet loss.")
    parser.add_argument("--aug_data_path", type=str, default=None)

    # flipda
    parser.add_argument("--search_type",default='baseline',type=str) #jing
    parser.add_argument("--aug_ids", default=[0,2], type=int, nargs='+',) #jing
    parser.add_argument("--filter_pattern",default=-1,type=int) #jing
    parser.add_argument("--fixla_ratio",default='[[0.1,0.1],[0.1,0.1]]',type=str)
    parser.add_argument('--pattern_iter_output_dir', type=str)
    """

    args = parser.parse_args()
    return args