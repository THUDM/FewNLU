import re
import math
import numpy as np
import torch
from torch import nn
import random
from transformers import __version__ as transformers_version
from methods.base_model import BaseModel
from functools import reduce

import log
logger = log.get_logger('root')

class AdaPETModel(BaseModel):
    def __init__(self, config, tokenizer, pvp):
        super(AdaPETModel, self).__init__(config, tokenizer, "mlm")
        self.config = config
        self.tokenizer = tokenizer
        self.pvp = pvp

        self.num_lbl = len(self.config.label_list)
        self.lbl_idx_lkup = nn.Embedding.from_pretrained(torch.eye(self.num_lbl)).to(config.device)
        assert config.use_cloze == True and config.use_continuous_prompt == False
        self.loss = nn.BCELoss(reduction="none")
        self.use_dropout=False      

    def train_step(self, labeled_batch, extra_batch, alpha, **kwargs):
        if 'use_dropout' in kwargs and kwargs['use_dropout']==True:
            self.use_dropout=True
        else:
            self.use_dropout=False
        num_lbl = len(self.config.label_list)
        lbl_idx_lkup = nn.Embedding.from_pretrained(torch.eye(num_lbl)).to(labeled_batch['input_ids'].device)

        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']

        if self.config.task_name in ["wsc", "copa"]:

            lbl_logits, lbl_ids, _ = self.get_multilbl_logits(labeled_batch)
            # [bs, num_lbl, max_num_lbl_tok]
            if self.config.task_name == "wsc":
                reshape_lbl_logits = lbl_logits.reshape(-1)  # [bs * num_lbl * max_num_lbl_tok]
                reshape_lbl = torch.ones_like(reshape_lbl_logits)
                real_mask = lbl_logits > 0
            else:
                # Removing tokens that are common across choices
                same_words_ids = torch.stack(
                    [reduce(lambda x, y: (x == y) * y, lbl_logit) for lbl_logit in lbl_logits], dim=0)
                mask_same_words = (1 - (same_words_ids > 0).long()).repeat(1, lbl_logits.shape[1], 1)
                # [bs, num_lbl, max_num_lbl_tok]
                real_mask = mask_same_words * (lbl_ids > 0)

                # Applying the mask to the lbl_logits
                lbl_logits = lbl_logits * mask_same_words  # [bs, num_lbl, max_num_lbl_tok]
                reshape_lbl_logits = lbl_logits.reshape(-1)  # [bs * num_lbl * max_num_lbl_tok]

                with torch.no_grad():
                    lkup_lbl = lbl_idx_lkup(labels.long())  # [bs, num_lbl]
                reshape_lbl = lkup_lbl[:, :, None].repeat(1, 1, self.config.max_num_lbl_tok).reshape(-1)

            full_sup_loss = self.loss(reshape_lbl_logits, reshape_lbl)  # [bs * num_lbl * max_num_lbl_tok]
            full_sup_loss = full_sup_loss.reshape(lbl_logits.shape)
            pet_disc_loss = torch.sum(full_sup_loss * real_mask) / torch.sum(real_mask)
        else:
            lbl_logits = self.get_single_logits(labeled_batch)
            reshape_lbl_logits = lbl_logits.reshape(-1)
            with torch.no_grad():
                lkup_lbl = lbl_idx_lkup(labels)
            reshape_lbl = lkup_lbl.reshape(-1)  # [bs*num_lbl]
            pet_disc_loss = torch.mean(nn.BCELoss()(reshape_lbl_logits, reshape_lbl))

        pet_disc_loss.backward()

        ### PET MLM loss
        # lm_inputs = self.generate_default_inputs(extra_batch)
        
        input_ids = extra_batch["input_ids"]
        mask_input_ids = extra_batch["mask_input_ids"]
        tgt = extra_batch["tgt"]
        correct_vocab_prob=self.get_pet_mlm_logits(input_ids,mask_input_ids)
        max_seq_len = correct_vocab_prob.shape[1]

        # mlm_loss = nn.BCELoss(reduce=False, reduction=None)
        # import pdb 
        # pdb.set_trace()
        full_loss = self.loss(correct_vocab_prob,
                             tgt[:, None].repeat(1, max_seq_len).float())
        mask_loss = input_ids != mask_input_ids
        pet_mlm_loss = torch.sum(full_loss * mask_loss.float()) / torch.max(torch.sum(mask_loss), torch.tensor(1).to(full_loss.device))
        # import pdb 
        # pdb.set_trace()
        # print(pet_disc_loss,pet_mlm_loss.clone().detach())


        if self.config.adapet_balance_alpha==-1:
            loss = pet_disc_loss.clone().detach() + pet_mlm_loss
        else:
            loss = (1-self.config.adapet_balance_alpha)*pet_disc_loss.clone().detach() + self.config.adapet_balance_alpha*pet_mlm_loss
        return loss

    def get_pet_mlm_logits(self, input_ids, masked_input_ids):
        '''
        Get logits for PET MLM objective

        :param input_ids: [bs, max_seq_len]
        :param masked_input_ids: [bs, max_seq_len]
        :return:
        '''
        pet_mask_rep = self(masked_input_ids, (masked_input_ids > 0).long(),use_dropout=self.use_dropout)[0]  # [bs, max_seq_len, vocab_size]
        pet_mask_rep_vocab_prob = pet_mask_rep.softmax(dim=-1)  # [bs, max_num_lbl_tok, vocab_size]

        pet_mask_rep_correct_vocab_prob = torch.gather(pet_mask_rep_vocab_prob, 2, input_ids[:,:,None]).squeeze(2) # [bs, max_seq_len]

        return pet_mask_rep_correct_vocab_prob

    def get_single_logits(self, labeled_batch):

        inputs = self.generate_default_inputs(labeled_batch)
        inputs['use_dropout']=self.use_dropout
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']

        # TODO: only for single-token task
        # logger.warning("only for single-token task")

        bs = labeled_batch["input_ids"].shape[0]
        max_num_lbl_tok = 1
        lbl_ids = np.ones((self.num_lbl, max_num_lbl_tok)) * self.tokenizer.pad_token_id
        list_lbl = [self.pvp.verbalize(item)[0] for item in self.config.label_list]
        for i, lbl in enumerate(list_lbl):
            i_lbl_ids = self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
            assert len(i_lbl_ids) == 1
            lbl_ids[i, :len(i_lbl_ids)] = i_lbl_ids
        lbl_ids = torch.tensor(lbl_ids).to(labeled_batch['input_ids'].device).long()

        list_mask_idx = np.ones((bs, 1)) * (self.config.max_seq_length-1)
        for bidx, idx in enumerate(mlm_labels):
            cur_mask_idx = mlm_labels[bidx].tolist().index(1)
            list_mask_idx[bidx, 0] = cur_mask_idx
        list_mask_idx = torch.tensor(list_mask_idx).to(labeled_batch['input_ids'].device)

        init_mask_idx_lkup = torch.cat([torch.eye(self.config.max_seq_length), torch.zeros((1, self.config.max_seq_length))], dim=0)
        mask_idx_lkup = nn.Embedding.from_pretrained(init_mask_idx_lkup).to(labeled_batch['input_ids'].device)
        with torch.no_grad():
            mask_idx_emb = mask_idx_lkup(list_mask_idx.long())

        pet_logits = self(**inputs)[0]
        pet_mask_logits = torch.matmul(mask_idx_emb[:, :, None, :], pet_logits[:, None, :, :]).squeeze(2)
        pet_mask_rep_vocab_prob = pet_mask_logits.softmax(dim=2)
        bs_by_max_num_lbl_tok = list(pet_mask_logits.shape[:2])

        mask_prob = torch.gather(pet_mask_rep_vocab_prob, 2, lbl_ids.view(1, -1).unsqueeze(1).repeat(bs_by_max_num_lbl_tok + [1]))
        mask_prob = mask_prob.transpose(1, 2)  # [bs,  max_num_lbl_tok*num_lbl, num_lbl_tok]
        mask_prob = mask_prob.reshape(bs, self.num_lbl, max_num_lbl_tok, max_num_lbl_tok)
        mask_diag_prob = torch.diagonal(mask_prob, dim1=2, dim2=3)  # [bs, num_lbl, num_lbl_tok]

        lbl_prob = torch.sum(mask_diag_prob, dim=2)
        return lbl_prob

    def get_multilbl_logits(self,labeled_batch):
        pet_mask_ids = labeled_batch["input_ids"]
        bs = pet_mask_ids.shape[0]

        if self.config.task_name=='copa':
            list_mask_idx = np.ones((bs, 2, self.config.max_num_lbl_tok)) * (self.config.max_seq_length-1)
            for bidx, idx in enumerate(labeled_batch['input_ids']):
                indexs=[i for (i,x) in enumerate(labeled_batch['input_ids'][bidx].detach().cpu().numpy()) if x==self.tokenizer.mask_token_id]
                list_mask_idx[bidx, 0, :len(indexs)] = torch.tensor(indexs,dtype=torch.long)
                list_mask_idx[bidx, 1, :len(indexs)] = torch.tensor(indexs,dtype=torch.long)
            mask_idx = torch.tensor(list_mask_idx).to(labeled_batch['input_ids'].device)
            batch_list_lbl=[[x,y] for (x,y) in zip(labeled_batch['pure_choice1_token_ids'],labeled_batch['pure_choice2_token_ids'])]
        elif self.config.task_name=='wsc':
            list_mask_idx = np.ones((bs, 1, self.config.max_num_lbl_tok)) * (self.config.max_seq_length-1)
            for bidx, idx in enumerate(labeled_batch['input_ids']):
                indexs=[i for (i,x) in enumerate(labeled_batch['input_ids'][bidx].detach().cpu().numpy()) if x==self.tokenizer.mask_token_id]
                list_mask_idx[bidx, 0, :len(indexs)] = torch.tensor(indexs,dtype=torch.long)
            mask_idx = torch.tensor(list_mask_idx).to(labeled_batch['input_ids'].device)
            batch_list_lbl=labeled_batch['padded_target']

        num_lbl = mask_idx.shape[1]
        lbl_ids = np.zeros((bs, num_lbl, self.config.max_num_lbl_tok))  # [bs, num_lbl, max_num_lbl_tok]

        # Get lbl ids for multi token labels
        for i, list_lbl in enumerate(batch_list_lbl):
            if self.config.task_name=='wsc':
                lbl_ids[i,0,:len(list_lbl)]=list_lbl.detach().cpu().numpy()
            else:
                for j, lbl in enumerate(list_lbl):
                    lbl=lbl[:self.config.max_num_lbl_tok]
                    lbl_ids[i,j,:len(lbl)]=lbl.detach().cpu().numpy()

        lbl_ids = torch.from_numpy(lbl_ids).to(labeled_batch['input_ids'].device)
        # import pdb 
        # pdb.set_trace()
        # Get probability for each vocab token at the mask position
        pet_logits = self(pet_mask_ids, (pet_mask_ids > 0).long(),use_dropout=self.use_dropout)[0]  # [bs, max_seq_len, vocab_size]
        vs = pet_logits.shape[-1]
        mask_idx = mask_idx.reshape(bs, num_lbl * self.config.max_num_lbl_tok)
        pet_rep_mask_ids_logit = torch.gather(pet_logits, 1, mask_idx[:, :, None].repeat(1, 1, vs).long())  # [bs, num_lbl * max_num_lbl_tok, vs]
        pet_rep_mask_ids_logit = pet_rep_mask_ids_logit.reshape(bs, num_lbl, self.config.max_num_lbl_tok, vs)  # [bs, num_lbl, max_num_lbl_tok, vs]
        pet_rep_mask_ids_prob = pet_rep_mask_ids_logit.softmax(dim=-1)

        # Compute logit for the lbl tokens at the mask position
        lbl_ids_expd = lbl_ids[..., None]  # [bs, num_lbl, max_num_lbl_tok, 1]
        pet_rep_mask_ids_lbl_logit = torch.gather(pet_rep_mask_ids_prob, 3, lbl_ids_expd.long()).squeeze(
            3)  # [bs, num_lbl, max_num_lbl_tok]

        if self.config.task_name=="wsc":
            masked_pet_rep_mask_ids_lbl_logit = pet_rep_mask_ids_lbl_logit * (
                        mask_idx != (pet_mask_ids.shape[-1] - 1)).unsqueeze(1).long()
        else:
            masked_pet_rep_mask_ids_lbl_logit = pet_rep_mask_ids_lbl_logit * (lbl_ids > 0).long()
        return masked_pet_rep_mask_ids_lbl_logit, lbl_ids, None

    def get_eval_wsc_logits(self, labeled_batch):
        # import pdb 
        # pdb.set_trace()
        pet_mask_ids = labeled_batch["input_ids"]
        bs = pet_mask_ids.shape[0]
        assert bs==1, f"Only implemented with batch size 1 for evaluating WSC"
        batch_list_lbl=[[self.tokenizer.decode([y for y in x if y!=0]) for x in labeled_batch['padded_target']]]
        # Assume batch size 0
        list_lbl = batch_list_lbl[0]
        mask_idx = [i for (i,x) in enumerate(labeled_batch['mlm_labels'][0].detach().cpu().numpy()) if x==1]

        while True:
            mask_positions = [
                idx for idx, input_id in enumerate(pet_mask_ids[0]) if input_id == self.tokenizer.mask_token_id
            ]
            if not mask_positions:  # there are no masks left to process, we are doneå
                input_ids = pet_mask_ids[0].detach().cpu().tolist()
                output_actual = self.tokenizer.decode([
                    input_id for idx, input_id in enumerate(input_ids)
                    if idx in mask_idx and input_id not in self.tokenizer.all_special_ids
                ])

                output_expected = list_lbl[0]

                # transform both outputs as described in the T5 paper
                output_actual = output_actual.lower().strip()
                output_actual = [w for w in re.split('[^a-zA-Z]', output_actual) if w]
                output_expected = output_expected.lower().strip()
                output_expected = [w for w in re.split('[^a-zA-Z]', output_expected) if w]

                # compare outputs
                if all(x in output_expected for x in output_actual) or all(
                        x in output_actual for x in output_expected):
                    return torch.tensor([[0, 1]])
                return torch.tensor([[1, 0]])

            outputs = self(pet_mask_ids, (pet_mask_ids > 0).long(),use_dropout=self.use_dropout)
            next_token_logits = outputs[0] # [batch_size, seq_len, vocab_size]
            ### TODO
            next_token_logits[:, :, 128000 :] = (-1) * math.inf
            next_token_logits = next_token_logits.softmax(dim=2)
            next_token_logits = next_token_logits[0].detach().cpu().numpy()

            most_confident = ()
            most_confident_score = -1

            for mask_position in mask_positions:
                ntl = next_token_logits[mask_position]
                top_token_id = np.argmax(ntl)
                top_score = ntl[top_token_id]
                if top_score > most_confident_score:
                    most_confident_score = top_score
                    most_confident = (mask_position, top_token_id)

            pet_mask_ids[0][most_confident[0]] = most_confident[1]

    def get_eval_multilbl_logits(self, labeled_batch):
        pet_mask_ids = torch.cat([labeled_batch["input_ids1"],labeled_batch["input_ids2"].clone()],dim=0)
        bs = 1
        
        mask_idx=[[i for (i,x) in enumerate(labeled_batch['input_ids1'][0].detach().cpu().numpy()) if x==self.tokenizer.mask_token_id],
        [i for (i,x) in enumerate(labeled_batch['input_ids2'][0].detach().cpu().numpy()) if x==self.tokenizer.mask_token_id]]
        # mask_idx=[[i for (i,x) in enumerate(labeled_batch['mlm_labels'][0].detach().cpu().numpy()) if x==1],
        #     [i for (i,x) in enumerate(labeled_batch['mlm_labels'][1].detach().cpu().numpy()) if x==1]]
        # print(mask_idx)
        if self.config.task_name=='copa':
            batch_list_lbl=[[x,y] for (x,y) in zip(labeled_batch['pure_choice1_token_ids'],labeled_batch['pure_choice2_token_ids'])]
        elif self.config.task_name=='wsc':
            batch_list_lbl=labeled_batch['padded_target']

        log_probs = []
        # Assume batch size 0
        list_lbl = batch_list_lbl[0]

        for idx, lbl_ids in enumerate(list_lbl):
            # lbl_ids = self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
            log_probabilities = []
            lbl_ids=lbl_ids[:min(len(mask_idx[idx]),len(lbl_ids))]
            while True:
                masks = [(idx, tok_id) for idx, tok_id in zip(mask_idx[idx], lbl_ids) if tok_id != -100]
                if not masks:
                    break
                pet_rep = self(pet_mask_ids[idx:idx + 1], (pet_mask_ids[idx:idx + 1] > 0).long(),use_dropout=self.use_dropout)[
                    0]  # [bs, max_seq_len]
                next_token_logits = pet_rep.softmax(dim=-1)[
                    0]  # The last indexing operation gets rid of batch dimension

                # Only implementing the 'default' non-autoregressive strategy for now
                mask_pos, masked_id = None, None
                max_prob = None
                for m_pos, m_id in masks:
                    m_prob = next_token_logits[int(m_pos)][int(m_id)].item()
                    if max_prob is None or m_prob > max_prob:
                        max_prob = m_prob
                        mask_pos, masked_id = m_pos, m_id

                log_probabilities.append(math.log(max_prob))
                pet_mask_ids[idx][mask_pos] = masked_id
                tok_pos = mask_idx[idx].index(mask_pos)
                lbl_ids[tok_pos] = -100

            log_probs.append(sum(log_probabilities))

        return torch.tensor([log_probs])

    def eval_step(self, batch, **kwargs):
        if 'use_dropout' in kwargs and kwargs['use_dropout']==True:
           self.use_dropout=True
        else:
            self.use_dropout=False
        if self.config.task_name in "wsc":
            lbl_logits = self.get_eval_wsc_logits(batch)
        elif self.config.task_name == "copa":
            lbl_logits = self.get_eval_multilbl_logits(batch)
        else:
            lbl_logits = self.get_single_logits(batch)
        pred_lbl, lbl_logits = torch.argmax(lbl_logits, dim=1), lbl_logits
        return lbl_logits


    def generate_default_inputs(self, batch):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet', 'deberta']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

