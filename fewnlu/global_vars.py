from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification, AlbertForMaskedLM, \
    DebertaV2Config, DebertaV2Tokenizer, BertConfig, BertTokenizer, BertForSequenceClassification, BertForMaskedLM, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM

from modified_hf_models.modeling_deberta_v2 import DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification

# processor
TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
DEV32_SET = "dev32"
UNLABELED_SET = "unlabeled"
AUGMENTED_SET = "augmented"
SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, DEV32_SET, UNLABELED_SET, AUGMENTED_SET]

# dataset
DEFAULT_METRICS = ["acc"]


# models
SEQUENCE_CLASSIFIER_WRAPPER = "cls"
MLM_WRAPPER = "mlm"
MODEL_CLASSES = {
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification,
        MLM_WRAPPER: AlbertForMaskedLM,
    },
    'deberta': {
        'config': DebertaV2Config,
        'tokenizer': DebertaV2Tokenizer,
        MLM_WRAPPER: DebertaV2ForMaskedLM,
        SEQUENCE_CLASSIFIER_WRAPPER: DebertaV2ForSequenceClassification
    },
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification,
        MLM_WRAPPER: RobertaForMaskedLM
    },
}


# SELF_TRAINING_METHODS = ['ipet', 'noisy_student']
UNLABELED_BASED_METHODS = ['lm_training']



SEQUENCE_CLASSIFIER_WRAPPER = "cls"
MLM_WRAPPER = "mlm"
PLM_WRAPPER = "plm"
WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PLM_WRAPPER]



CONFIG_NAME = 'wrapper_config.json'
TRAIN_EVAL_CONFIG_NAME = 'train_eval_config.json'
