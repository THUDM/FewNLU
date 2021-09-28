from abc import abstractmethod

import torch
import torch.nn as nn
from transformers import AlbertForSequenceClassification, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    DebertaV2Tokenizer, DebertaV2Config, BertConfig, BertTokenizer, BertForSequenceClassification, BertForMaskedLM, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM

from modified_hf_models.modeling_deberta_v2 import DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification
import log
from data_utils.preprocessor import SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER
from global_vars import MODEL_CLASSES

logger = log.get_logger()

class DropoutWords(nn.Dropout2d): #Spatial Dropout
    def forward(self,x):
        x=x.unsqueeze(2)
        x=x.permute(0,3,2,1)
        x=super(DropoutWords,self).forward(x)
        x=x.permute(0,3,2,1)
        x=x.squeeze(2)
        return x

class BaseModel(torch.nn.Module):
    def __init__(self, config, tokenizer, wrapper_type):
        super(BaseModel, self).__init__()
        self.config = config
        self.tokenizer = tokenizer

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=len(config.label_list), finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)

        model_class = MODEL_CLASSES[self.config.model_type][wrapper_type]
        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)
        logger.info(" Base pretrained model Loaded.")

        if "deberta" in self.config.model_name_or_path and self.config.fix_deberta:
            logger.info(" DeBERTa model with fix_layers.")
            self.model.fix_layers()
        self.dropout = DropoutWords(config.dropout_rate)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                inputs_embeds=None, use_dropout=False, **kwargs):
        if inputs_embeds is not None:
            raw_embeds=inputs_embeds
        else:
            assert input_ids is not None
            raw_embeds = self.model.get_input_embeddings()(input_ids)  # [batch_size, seq_len, embed_size]
        if use_dropout==True:
            raw_embeds = self.dropout(raw_embeds)
        return self.model(inputs_embeds=raw_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          labels=labels, **kwargs)

    # def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
    #             inputs_embeds=None, is_training=True, **kwargs):
    #     return self.model(input_ids=input_ids,
    #                       attention_mask=attention_mask,
    #                       token_type_ids=token_type_ids,
    #                       labels=labels,
    #                       inputs_embeds=inputs_embeds,
    #                       **kwargs)

    @abstractmethod
    def train_step(self, batch, extra_batch, alpha, **_):
        pass

    @abstractmethod
    def eval_step(self, batch, **_):
        pass

    @abstractmethod
    def generate_default_inputs(self, batch):
        pass
