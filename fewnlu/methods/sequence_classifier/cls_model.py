from methods.base_model import BaseModel

import log
logger = log.get_logger('root')

class SequenceClassifierModel(BaseModel):

    def __init__(self, config, tokenizer, pvp=None):
        super(SequenceClassifierModel, self).__init__(config, tokenizer, "cls")
        assert config.use_cloze == False

    def train_step(self, batch, **_):
        inputs = self.generate_default_inputs(batch)
        inputs['labels'] = batch['labels']
        outputs = self.model(**inputs)
        return outputs[0]


    def eval_step(self, batch, **_):
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]

    def generate_default_inputs(self, batch):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet', 'deberta']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs