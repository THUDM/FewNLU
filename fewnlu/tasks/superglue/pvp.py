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
This file contains the different strategies to patternize data for all SuperGLUE tasks, including
direct concatenation, pattern-verbalizer pairs (PVPs), and ptuning PVPs.
"""


import string
from typing import List

from utils import InputExample, get_verbalization_ids
import log
from tasks.base_pvp import PVP, PVPOutputPattern

logger = log.get_logger()


class RtePVP(PVP):

    _is_multi_token = False

    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    MULTI_VERBALIZER={
        "not_entailment": ["No","false"],
        "entailment": ["Yes","true"]
    }

    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        else:
            return [1, 2, 3, 4, 5, 6, 8, 10]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)  # premise
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))  # hypothesis

        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:
            if self.pattern_id == 0 or self.pattern_id == 10:
                return ['"', text_b, '" ?'], [[self.mask_id], ', "', text_a, '"']
            elif self.pattern_id == 1 or self.pattern_id == 11:
                return [text_b, '?'], [[self.mask_id], ',', text_a]
            elif self.pattern_id == 2 or self.pattern_id == 12:
                return ['"', text_b, '" ?'], [[self.mask_id], '. "', text_a, '"']
            elif self.pattern_id == 3 or self.pattern_id == 13:
                return [text_b, '?'], [[self.mask_id], '.', text_a]
            elif self.pattern_id == 4 or self.pattern_id == 14:
                return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', [self.mask_id]], []
            elif self.pattern_id == 5 or self.pattern_id == 15:
                return [text_a, 'Question:', text_b, "?", "Answer:", [self.mask_id], "."], []

        else:
            if self.pattern_id == 1:
                return [text_a, 'Question:', text_b, "?", 1, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 2:
                return [text_a, 1, 'Question:', text_b, "?", 1, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 3:
                return [text_a, 1, 'Question:', text_b, "?", 2, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 4:
                return [text_a, 2, 'Question:', text_b, "?", 2, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 5:
                return [text_a, 2, 'Question:', text_b, "?", 3, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 6:
                return [text_a, 3, 'Question:', text_b, "?", 3, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 8:
                return [text_a, 4, 'Question:', text_b, "?", 4, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 10:
                return [text_a, 5, 'Question:', text_b, "?", 5, "Answer:", [self.mask_id], "."], []


    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []

        if self.use_continuous_prompt:
            return RtePVP.VERBALIZER[label]

        elif self.pattern_id == 4:
            return ['true'] if label == 'entailment' else ['false']
        elif self.pattern_id >= 10:
            return RtePVP.MULTI_VERBALIZER[label]
        else:
            return RtePVP.VERBALIZER[label]


class CbPVP(PVP):
    _is_multi_token = False
    VERBALIZER = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    MULTI_VERBALIZER = {
        "contradiction": ["No","false"],
        "entailment": ["Yes","true"],
        "neutral": ["Maybe","neither"]
    }

    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        else:
            return [1, 2, 3, 4, 5, 6, 8, 10]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)  # premise
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))  # hypothesis
        assert self.pattern_id in self.available_patterns()
        if not self.use_cloze:
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:
            if self.pattern_id == 0 or self.pattern_id==10:
                return ['"', text_b, '" ?'], [[self.mask_id], ', "', text_a, '"']
            elif self.pattern_id == 1 or self.pattern_id == 11:
                return [text_b, '?'], [[self.mask_id], ',', text_a]
            elif self.pattern_id == 2 or self.pattern_id == 12:
                return ['"', text_b, '" ?'], [[self.mask_id], '. "', text_a, '"']
            elif self.pattern_id == 3 or self.pattern_id == 13:
                return [text_b, '?'], [[self.mask_id], '.', text_a]
            elif self.pattern_id == 4 or self.pattern_id == 14:
                return [text_a, ' question: ', self.shortenable(example.text_b), ' true, false or neither? answer:', [self.mask_id]], []
            elif self.pattern_id == 5 or self.pattern_id == 15:
                return [text_a, 'Question:', text_b, "?", "Answer:", [self.mask_id], "."], []
        else:
            if self.pattern_id == 1:
                return [text_a, 'Question:', text_b, "?", 1, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 2:
                return [text_a, 1, 'Question:', text_b, "?", 1, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 3:
                return [text_a, 1, 'Question:', text_b, "?", 2, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 4:
                return [text_a, 2, 'Question:', text_b, "?", 2, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 5:
                return [text_a, 2, 'Question:', text_b, "?", 3, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 6:
                return [text_a, 3, 'Question:', text_b, "?", 3, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 8:
                return [text_a, 4, 'Question:', text_b, "?", 4, "Answer:", [self.mask_id], "."], []
            elif self.pattern_id == 10:
                return [text_a, 5, 'Question:', text_b, "?", 5, "Answer:", [self.mask_id], "."], []


    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []
        if self.use_continuous_prompt:
            # return ['true'] if label == 'entailment' else ['false'] if label == 'contradiction' else ['neither']
            return CbPVP.VERBALIZER[label]
        elif self.pattern_id == 4:
            return ['true'] if label == 'entailment' else ['false'] if label == 'contradiction' else ['neither']
        elif self.pattern_id == 10:
            return CbPVP.MULTI_VERBALIZER[label]
        else:
            return CbPVP.VERBALIZER[label]


class CopaPVP(PVP):
    _is_multi_token = True
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1]
        else:
            return [1,2,3,4,5,6,8,10]


    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        premise = self.remove_final_punc(self.shortenable(example.text_a))
        choice1 = self.remove_final_punc(self.lowercase_first(example.meta['choice1']))
        choice2 = self.remove_final_punc(self.lowercase_first(example.meta['choice2']))
        question = example.meta['question']
        joiner = 'because' if question == 'cause' else ', so'
        assert question in ['cause', 'effect']

        example.meta['choice1'], example.meta['choice2'] = choice1, choice2
        num_masks = max(len(get_verbalization_ids(c, self.tokenizer, False)) for c in [choice1, choice2])

        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            text_a = ' '.join([self.remove_final_punc(example.text_a), joiner, self.lowercase_first(example.meta['choice1'])])
            text_b = ' '.join([self.remove_final_punc(example.text_a), joiner, self.lowercase_first(example.meta['choice2'])])
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:

            if self.pattern_id == 0:
                return ['"', choice1, '" or "', choice2, '"?', premise, joiner, [self.mask_id] * num_masks, '.'], []
            elif self.pattern_id == 1:
                return [choice1, 'or', choice2, '?', premise, joiner, [self.mask_id] * num_masks, '.'], []

        else:

            if self.pattern_id == 1:
                # return ['"', choice1, '" or "', choice2, '"?', 1, premise, joiner, [self.mask_id] * num_masks, '.'], []
                return ['"', choice1, '" or "', choice2, '"?', premise, joiner, 1, [self.mask_id] * num_masks, '.'], []
            elif self.pattern_id == 2:
                return ['"', choice1, '" or "', choice2, '"?', 2, premise, joiner, [self.mask_id] * num_masks, '.'], []
            elif self.pattern_id == 3:
                return ['"', choice1, '" or "', choice2, '"?', 3, premise, joiner, [self.mask_id] * num_masks, '.'], []
            elif self.pattern_id == 4:
                # return ['"', choice1, '" or "', choice2, '"?', 2, premise, joiner, 2, [self.mask_id] * num_masks, '.'], []
                return ['"', choice1, '" or "', choice2, '"?', premise, joiner, 4, [self.mask_id] * num_masks,
                        '.'], []
            elif self.pattern_id == 5:
                return ['"', choice1, '" or "', choice2, '"?', 3, premise, joiner, 2, [self.mask_id] * num_masks, '.'], []
            elif self.pattern_id == 6:
                return ['"', choice1, '" or "', choice2, '"?', 3, premise, joiner, 3, [self.mask_id] * num_masks, '.'], []



    def verbalize(self, label) -> List[str]:
        return []


class WscPVP(PVP):
    _is_multi_token = True
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0,1,2]
        else:
            return [1,2,3,4,5,6,8]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        pronoun = example.meta['span2_text']
        pronoun_idx = example.meta['span2_index']

        words_a = example.text_a.split()
        words_a[pronoun_idx] = '*' + words_a[pronoun_idx] + '*'
        text_a = ' '.join(words_a)
        text_a = self.shortenable(text_a)

        target = example.meta['span1_text']
        num_pad = self.rng.randint(0, 3) if 'train' in example.guid else 1
        num_masks = len(get_verbalization_ids(target, self.tokenizer, force_single_token=False)) + num_pad
        masks = [self.mask_id] * num_masks

        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            text_b = target
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:

            if self.pattern_id == 0:
                return [text_a, "The pronoun '*" + pronoun + "*' refers to", masks, '.'], []
            elif self.pattern_id == 1:
                return [text_a, "In the previous sentence, the pronoun '*" + pronoun + "*' refers to", masks, '.'], []
            elif self.pattern_id == 2:
                return [text_a,
                        "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to? Answer: ", masks, '.'], []
        else:
            if self.pattern_id == 1:
                return [text_a, 1,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to? Answer: ", masks,
                 '.'], []
            elif self.pattern_id == 2:
                return [text_a, 2,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to? Answer: ", masks,
                 '.'], []
            elif self.pattern_id == 3:
                return [text_a, 3,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to? Answer: ", masks,
                 '.'], []
            elif self.pattern_id == 4:
                return [text_a, 4,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to? Answer: ", masks,
                 '.'], []
            elif self.pattern_id == 5:
                return [text_a, 3,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to?", 2, "Answer: ",
                        masks,
                 '.'], []
            elif self.pattern_id == 6:
                return [text_a, 3,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to?", 3,
                        "Answer: ", masks, '.'], []
            elif self.pattern_id == 8:
                return [text_a, 4,
                 "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to?",4, "Answer: ",
                        masks,
                 '.'], []

    def verbalize(self, label) -> List[str]:
        return []



class BoolQPVP(PVP):
    _is_multi_token = False
    VERBALIZER_A = {
        "False": ["No"],
        "True": ["Yes"]
    }

    VERBALIZER_B = {
        "False": ["false"],
        "True": ["true"]
    }

    MULTI_VERBALIZER={
        "False": ["No","false"],
        "True": ["Yes","true"]
    }
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        else:
            return [1, 2, 3, 4, 5, 6, 8, 10]


    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        passage = self.shortenable(example.text_a)
        question = self.shortenable(self.remove_final_punc(example.text_b))

        assert self.pattern_id in self.available_patterns()
        if not self.use_cloze:
            return [passage], [question]

        elif not self.use_continuous_prompt:
            if self.pattern_id < 2 or self.pattern_id % 10 <2:
                return [passage, '. Question: ', question, '? Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id < 4 or self.pattern_id % 10 <4:
                return [passage, '. Based on the previous passage, ', question, '?', [self.mask_id], '.'], []
            else:
                return ['Based on the following passage, ', question, '?', [self.mask_id], '.', passage], []
        else:
            if self.pattern_id == 1:
                return [passage, '. Question: ', question, '?', 1, 'Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id == 2:
                return [passage, '.', 1, 'Question: ', question, '?', 1, 'Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id == 3:
                return [passage, '.', 1, 'Question: ', question, '?', 2, 'Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id == 4:
                return [passage, '.', 2 ,'Question: ', question, '?', 2, 'Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id == 6:
                return [passage, '.', 3, 'Question: ', question, '?', 3, 'Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id == 8:
                return [passage, '.', 4, 'Question: ', question, '?', 4, 'Answer: ', [self.mask_id], '.'], []
            elif self.pattern_id == 10:
                return [passage, '.', 5, 'Question: ', question, '?', 5, 'Answer: ', [self.mask_id], '.'], []

    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []
        if self.use_continuous_prompt:
            return BoolQPVP.VERBALIZER_B[label]
            # return BoolQPVP.VERBALIZER_A[label]
        elif self.pattern_id >= 10:
            return BoolQPVP.MULTI_VERBALIZER[label]
        elif self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return BoolQPVP.VERBALIZER_A[label]
        else:
            return BoolQPVP.VERBALIZER_B[label]


class MultiRcPVP(PVP):
    _is_multi_token = False
    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    }

    MULTI_VERBALIZER={
        "0": ["No","False"],
        "1": ["Yes","True"]
    }

    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3, 10, 11, 12, 13]
        else:
            return [1, 2, 3, 4, 6, 9]


    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        passage = self.shortenable(example.text_a)
        question = self.remove_final_punc(example.text_b)
        answer = example.meta['answer']
        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            text_a = passage
            text_b = ' '.join([example.text_b, self.tokenizer.sep_token, answer])
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:

            if self.pattern_id == 0 or self.pattern_id==10:
                return [passage, '. Question: ', question, '? Is it ', answer, '?', [self.mask_id], '.'], []
            elif self.pattern_id == 1 or self.pattern_id==11:
                return [passage, '. Question: ', question, '? Is the correct answer "', answer, '"?', [self.mask_id], '.'], []
            elif self.pattern_id == 2 or self.pattern_id==12:
                return [passage, '. Based on the previous passage, ', question, '? Is "', answer, '" a correct answer?', [self.mask_id], '.'], []
            elif self.pattern_id == 3 or self.pattern_id==13:
                return [passage, question, '- [', [self.mask_id], ']', answer], []

        else:
            if self.pattern_id == 1:
                return [passage, '. Question: ', question, '? Is it ', answer, '?', 1, [self.mask_id], '.'], []
            elif self.pattern_id == 2:
                return [passage, '. Question: ', question, '?', 1, 'Is it ', answer, '?', 1, [self.mask_id], '.'], []
            elif self.pattern_id == 3:
                return [passage, '.',1, 'Question: ', question, '?', 1, 'Is it ', answer, '?', 1, [self.mask_id], '.'], []
            elif self.pattern_id == 4:
                return [passage, '.', 1, 'Question: ', question, '?', 1, 'Is it ', answer, '?', 2, [self.mask_id],'.'], []
            elif self.pattern_id == 5:
                return [passage, '.', 1, 'Question: ', question, '?', 2, 'Is it ', answer, '?', 2, [self.mask_id],'.'], []
            elif self.pattern_id == 6:
                return [passage, '.', 2, 'Question: ', question, '?', 2, 'Is it ', answer, '?', 2, [self.mask_id],'.'], []
            elif self.pattern_id == 9:
                return [passage, '.',3, 'Question: ', question, '?', 3, 'Is it ', answer, '?', 3, [self.mask_id], '.'], []


    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []
        if self.use_continuous_prompt:
            return MultiRcPVP.VERBALIZER[label]
        elif self.pattern_id >=10:
            return MultiRcPVP.MULTI_VERBALIZER[label]
        elif self.pattern_id == 3:
            return ['False'] if label == "0" else ['True']
        else:
            return MultiRcPVP.VERBALIZER[label]


class WicPVP(PVP):
    _is_multi_token = False
    VERBALIZER_A = {
        "F": ["No"],
        "T": ["Yes"]
    }
    VERBALIZER_B = {
        "F": ["2"],
        "T": ["b"]
    }

    MULTI_VERBALIZER = {
        "F": ["No","False"],
        "T": ["Yes","True"]
    }
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 10, 20, 30]
        else:
            return [1, 2, 3, 4, 5, 6, 7, 8, 10]


    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        word = example.meta['word']
        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            text_a = self.shortenable(example.meta['word'] + ': ' + example.text_a)
            text_b = self.shortenable(example.text_b)
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:

            if self.pattern_id == 0 or self.pattern_id == 10:
                return ['"', text_a, '" / "', text_b, '" Similar sense of "' + word + '"?', [self.mask_id], '.'], []
            elif self.pattern_id == 1 or self.pattern_id == 11:
                return [text_a, text_b, 'Does ' + word + ' have the same meaning in both sentences?', [self.mask_id]], []
            elif self.pattern_id == 2 or self.pattern_id == 12:
                return [word, ' . Sense (1) (a) "', text_a, '" (', [self.mask_id], ') "', text_b, '"'], []

        else:
            """
            if self.pattern_id == 1:
                # return [text_a, '[SEP]', text_b , 1, word + '?', [self.mask_id]], []
                return [text_a, text_b, 'Does ' + word + ' have the same meaning in both sentences?', 1, [self.mask_id]], []
            elif self.pattern_id == 2:
                # return [text_a, '[SEP]', text_b , 2, word + '?', [self.mask_id]], []
                return [text_a, text_b, 'Does ' + word + ' have the same meaning in both sentences?', 2, [self.mask_id]], []
            elif self.pattern_id == 3:
                # return [text_a, '[SEP]', text_b , 3, word + '?', [self.mask_id]], []
                return [text_a, text_b, 'Does ' + word + ' have the same meaning in both sentences?', 3, [self.mask_id]], []
            elif self.pattern_id == 4:
                # return [text_a, '[SEP]', text_b , 4, word + '?', [self.mask_id]], []
                return [text_a, text_b, 'Does ' + word + ' have the same meaning in both sentences?', 4, [self.mask_id]], []
            elif self.pattern_id == 5:
                # return [text_a, '[SEP]', text_b , 3, word + '?', 2, [self.mask_id]], []
                return [text_a, text_b, 2, 'Does ' + word + ' have the same meaning in both sentences?', 3, [self.mask_id]], []
            elif self.pattern_id == 6:
                # return [text_a, '[SEP]', text_b , 3, word + '?', 3, [self.mask_id]], []
                return [text_a, 2, text_b, 2, 'Does ' + word + ' have the same meaning in both sentences?', 2,
                        [self.mask_id]], []
            """

            if self.pattern_id == 1:
                return [word, ' . Sense (1) (a) "', text_a, '" (', [self.mask_id], 1, ') "', text_b, '"'], []
            elif self.pattern_id == 2:
                return [word, ' . Sense (1) (a) "', text_a, '" (', 1, [self.mask_id], 1, ') "', text_b, '"'], []
            elif self.pattern_id == 4:
                return [word, ' . Sense (1) (a) "', text_a, '" (', 2, [self.mask_id], 2, ') "', text_b, '"'], []
            elif self.pattern_id == 6:
                return [word, ' . Sense (1) (a) "', text_a, '" (', 3, [self.mask_id], 3, ') "', text_b, '"'], []
            elif self.pattern_id == 8:
                return [word, ' . Sense (1) (a) "', text_a, '" (', 4, [self.mask_id], 4, ') "', text_b, '"'], []
            elif self.pattern_id == 10:
                return [word, ' . Sense (1) (a) "', text_a, '" (', 5, [self.mask_id], 5, ') "', text_b, '"'], []



    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []
        if self.use_continuous_prompt:
            return WicPVP.VERBALIZER_B[label]
            # return WicPVP.VERBALIZER_A[label]
        elif self.pattern_id >=10:
            return WicPVP.MULTI_VERBALIZER[label]
        elif self.pattern_id == 2:
            return WicPVP.VERBALIZER_B[label]
        else:
            return WicPVP.VERBALIZER_A[label]


class RecordPVP(PVP):
    _is_multi_token = True
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0]
        else:
            return [1,2,3]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        premise = self.shortenable(example.text_a)
        assert '@placeholder' in example.text_b, f'question "{example.text_b}" does not contain a @placeholder token'
        choices = example.meta['candidates']
        num_masks = max(len(get_verbalization_ids(c, self.tokenizer, False)) for c in choices)
        # question = example.text_b.replace('@placeholder', self.mask * num_masks)
        # assert self.mask in question
        question_a, question_b = example.text_b.split('@placeholder')

        assert self.pattern_id in self.available_patterns()
        if not self.use_cloze:
            raise NotImplementedError

        elif not self.use_continuous_prompt:
            if self.pattern_id == 0:
                return [premise, " " + question_a.rstrip(), [self.mask_id] * num_masks, question_b], []
        else:
            if self.pattern_id == 1:
                return [premise, 1, " " + question_a.rstrip(), [self.mask_id] * num_masks, question_b], []
            elif self.pattern_id == 2:
                return [premise, 2, " " + question_a.rstrip(), [self.mask_id] * num_masks, question_b], []
            elif self.pattern_id == 3:
                return [premise, 3, " " + question_a.rstrip(), [self.mask_id] * num_masks, question_b], []

    def verbalize(self, label) -> List[str]:
        return []



SUPERGLUE_PVPS = {
    'rte': RtePVP,
    'wic': WicPVP,
    'cb': CbPVP,
    'wsc': WscPVP,
    'boolq': BoolQPVP,
    'copa': CopaPVP,
    'multirc': MultiRcPVP,
    'record': RecordPVP
}


SUPERGLUE_METRICS = {
    "cb":       ["acc", "f1-macro"],
    "multirc":  ["acc", "f1", "em"],
    "record":   ["acc", "f1"],
    "boolq":    ["acc"],
    "rte":      ["acc"],
    "wic":      ["acc"],
    "copa":     ["acc"],
    "wsc":      ["acc"],
}