import torch

from methods.base_model import BaseModel

import log
logger = log.get_logger('root')

class LMTrainingModel(BaseModel):
    def __init__(self, config, tokenizer, pvp):
        super(LMTrainingModel, self).__init__(config, tokenizer, "mlm")
        self.config = config
        self.tokenizer = tokenizer
        self.pvp = pvp
        assert config.use_cloze == True and config.use_continuous_prompt == False

    def train_step(self, batch, extra_batch, alpha, **_):

        inputs = self.generate_default_inputs(batch)
        mlm_labels, labels = batch['mlm_labels'], batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = self.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = torch.nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        lm_inputs = self.generate_default_inputs(extra_batch)
        # lm_inputs['masked_lm_labels'] = extra_batch['mlm_labels']
        lm_inputs['labels'] = extra_batch['mlm_labels']
        lm_loss = self.model(**lm_inputs)[0]
        loss = alpha * loss + (1 - alpha) * lm_loss
        return loss


    def eval_step(self, batch, **_):
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])


    def generate_default_inputs(self, batch):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet', 'deberta']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs