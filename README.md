# FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding

## Introduction
Few-shot natural language understanding has attracted much recent attention. However, prior methods have been 
evaluated under a diverse set of protocols, which hinders fair comparison and measuring progress of the field. It is 
quested for a converged evaluation protocol as well as a general toolkit for few-shot NLU. 

FewNLU is an integrated toolkit designed for few-shot natural language understanding (Few-Shot NLU). It contains implementations of a number of state-of-the-art methods and data processing, a standard training procedure and most importantly, a justified evaluation framework for few-shot NLU proposed in the [FewNLU paper](https://arxiv.org/abs/2109.12742).  FewNLU also allows customizing new tasks and methods, and performing training and evaluation over them. Key features of FewNLU include: 
1. A justified evaluation framework with recommended data-split strategy for few-shot NLU.
2. A collection of state-of-the-Art methods for few-shot NLU.
3. Easy-to-Use customization of tasks and methods, which enables NLU to easily scale to a diverse range of future works.

#### Related Resources
1. Paper: [FewNLU: Benchmarking State-of-the-Art Methods for 
Few-Shot Natural Language Understanding](https://arxiv.org/abs/2109.12742).
2. Leaderboard: [https://fewnlu.github.io](https://fewnlu.github.io).
3. A new version of FewGLUE dataset: [Download](https://cloud.tsinghua.edu.cn/f/03b187bf3fff4a5fb1d1/?dl=1) and 
   [Homepage](https://paperswithcode.com/dataset/fewglue-64-labeled).
   
## Dataset
We construct the [FewGLUE_64_labeled dataset](https://paperswithcode.com/dataset/fewglue-64-labeled), which is a new 
version of [FewGLUE dataset](https://github.com/timoschick/fewglue). It contains a 64-sample training set, a 
development set (the original SuperGLUE development set), a test set, and an unlabeled set. It is constructed to 
facilitate the research of few-shot learning for natural language understanding tasks.

Compared with the original FewGLUE dataset, it differs in the number of labeled data examples in the training set, 
where the original FewGLUE has 32 trainining examples while FewGLUE_64_labeled has 64 labeled examples. Purposes for 
constructing a new version of FewGLUE dataset include:
1. To answer the questions that what is the best performance that few-shot learning can achieve and whether it is possible to further close the performance gap between few-shot learning and fully-supervised systems.
2. To explore to which degree the number of labeled training examples influences the few-shot performance.

Part of the FewGLUE_64_labeled dataset is based on the original 32-sample version of [FewGLUE](https://github.com/timoschick/fewglue). We collect them together in one package for the convenience of usage.
We appreciate all the contributors who made their dataset public.


## Leaderboard

We build the [FewNLU leaderboard](https://fewnlu.github.io) to facilitate few-shot NLU research based on the proposed evaluation framework. The 
proposed evaluation framework first compares all few-shot NLU methods on a common ground. The goal of the FewNLU leaderboard is to collect research works under the evaluation framework and to measure the true progress of the field constantly. We encourage researchers in this field to submit their own results obtained with FewNLU, with a link to the reproducible source codes attached.


## Toolkit

### Installation

Clone this repo:
```shell
git clone https://github.com/THUDM/FewNLU
cd FewNLU
```

To reproduce the exact results as is reported in the paper, please install exact the same version of dependencies by 
running:
```shell
pip install -r requirements.txt
```

If you use FewNLU to perform multiple comparative experiments, 
you are also supposed to keep exact the same environments as well as hardware devices of the same type.
***Several interesting observations (Just FYI, and we will keep updating if got detailed explanations)***  are that:
1. We perform the same experiments (SuperGLUE WSC task) respectively using one A100 GPU and one V100GPU, with the same code, hyper-params as well as exact the same dependencies,
and results vary a lot.
2. The version of Transformers(version 3.0.5 and version 4.5.1) affects differently on few-shot performance.
3. The version of Pytorch CUDA (10.5 and 11.0) affects few-shot performance.

### Direct Usage of Scripts
- Step1: Download either the original [32-sample version FewGLUE](https://github.com/timoschick/fewglue) dataset or our 
   [64-sample version FewGLUE](https://paperswithcode.com/dataset/fewglue-64-labeled) dataset. Set `DATA_DIR` with 
   your local data path. 
- Step2: Specify `SAVE_PATH` to your local path, indicating where to save model checkpoints.
- Step3: Run the following script (for example, run PET on the BoolQ task).
```
bash scripts/search_pet_devsplit.sh <task_name> <gpu_id> <model_type>
```
In the scripts, hyper-parameters to be considered for grid searching have been assigned appropriate search space.
For other base models, methods or tasks, you should specify your own search space.

Other adjustable key arguments include:
1. To choose among different few-shot methods, one should specify `--method`. Pre-defined methods include 
   standard sequence classification (`sequence_classifier`), pet (`pet`), ptuning (`ptuning`), adapet (`adapet`).
   You can also develop and customize your own new method by implementing a `Model` class. 
   Please refer to [Customizing your Own Methods](#Customizing your Own Methods) for more detailed instructions.
2. To choose among different training paradigms, one should specify `--arch`. Pre-defined training paradigms include 
   single-run paradigm (`default`), iPET (`ipet`) and Noisy Student (`noisy_student`)
3. To choose different tasks/datasets, one should specify `--dataset_name` (e.g., superglue) and `--task_name` (e.g.,
   rte).
You can also apply FewNLU to new NLU tasks. You can customize your own new tasks by 
   simply implementing a `DataProcessor` for loading data and a `PVP` for patternizing data inputs. 
   Please see [Customizing your Own Tasks](#Customizing your Own Tasks) for more detailed instructions.
3. To change your base pretrained models, one should specify `--model_type` (e.g., albert) and 
   `--model_name_or path` (e.g., microsoft/deberta-xxlarge-v2). Currently, FewNLU supports both
   bidirectional language models (e.g., bert-based models) and unidirectional models (e.g., gpt-based models).

### Customizing your Own Tasks 
To customize your own NLU tasks with FewNLU, you need to create your own dataset repository in [tasks](/tasks
), in which you define multiple tasks. For each task, you should implement a subclass of class `DataProcessor` in
 [/tasks
/base_processor.py
](/tasks/base_processor.py) and a subclass of class `PVP` in [/tasks/base_pvp.py](/tasks/base_pvp.py). Besides, you
 need to
 register your own dataset & tasks by defining `YOUR_METRICS`, `YOUR_PROCESSORS` and `YOUR_PVPS`.
 Here we take the MRPC task in [GLUE](/tasks/glue) dataset as an
  example. 

#### Step 1. Formalizing your Task with Patterns
We first need to design different types of patterns for the MRPC task.
The MRPC task is a paraphrase detection task, each data of which consists of two sentences and a label (`1` for
 paraphrases while `0` for non-paraphrases)
An example data from MRPC train set is as follows.
```
sentence1: Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.
sentence2: Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.
label: 1
```
(1) For standard classification fine-tuning, the pattern is designed by simply concatenating both sentences.
```
[sentence1][SEP][sentence2]
```
(2) When fine-tuning with manual discrete prompts, we design the patterns as follows. 
```
Does "[sentence1]" has the same meaning with "[sentence2]"? [MASK].
```
Accordingly, for verbalizers, the pretrained model predicts "Yes" for
 label `1` while "No" for label `0`. 
(3) For P-tuning which tunes pretrained models with continuous prompts, we design the patterns as follows.
Based on the designed manual discrete prompt, we insert several continuous prompt words (e.g., here we insert one
) into it.
```
Does "[sentence1]" has the same meaning with "[sentence2]"? [cont_prompt_word] [MASK].
```

#### Step 2. Implement subclass of `PVP`
The function of class `PVP` is to formalize inputs into different patterns. For FewNLU three different types of
 formalization strategies are provided for selection, including standard fine-tuning patterns, manual discrete patterns
  and
  continuous patterns (P-tuning).

To implement a subclass of `PVP`, you need to define a new class (e.g., class `MRPCPVP`) by inheriting superclass `PVP`.
Inside the class `MRPCPVP`, you should first specify the `VERBALIZER` variable, which defines the mapping between
 labels to verbalizers. 
Second, you should re-implement the abstract function `get_parts()` that returns a PVPOutputPattern object, such that
 pretrained models take it as inputs.
Third, you should re-implement the abstract function `verbalize()` that returns corresponding verbalizer word given
 a label.
 Take `MRPCPVP` as an example:
 ```rest
class MRPCPVP(PVP):
    VERBALIZER = {
        "1": ["Yes"],
        "0": ["No"]
    }
    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        if not self.use_cloze:
            return [text_a], [text_b]
        elif not self.use_continuous_prompt:
            return ["Does ", text_a, " has the same meaning with ", text_b, "? ", [self.mask_id], "."], []
        else:
            return ["Does ", text_a, " has the same meaning with ", text_b, "? ", 1, [self.mask_id], "."], []
           
    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []
        return MRPCPVP.VERBALIZER[label]
```
Note that `shortenable()` is used to mark that the segments can be truncated when exceeding the maximum sequence
 length. For P-tuning patterns, an integer is used to denote the number of continuous prompt words to be inserted here.

#### Step 3. Implement subclass of `DataProcessor`
The function of class `DataProcessor` is to provides methods for loading training, testing, development/dev32 and
 unlabeled examples for a given task.
 
To implement a subclass of class `DataProcessor`, you need to define a new class (e.g., `MRPCDataProcessor`) by
 inheriting class `DataProcessor`. 
 The new class `MRPCDataProcessor` needs to implement abstract methods `get_labels()` that returns the label list for
  current task, and `_create_examples()` which reads the data file according to its own format.
Here we take the `MRPCDataProcessor` as an example.
```
class MRPCDataProcessor(DataProcessor):
    def get_labels():
        return ["1", "0"]
def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        # TODO
        return examples
```

#### Step 4. Register your own Dataset & Tasks
After that, you should add dict-type variables `GLUE_METRICS` and `GLUE_PVPS` to [tasks/glue/pvp.py](tasks/glue/pvp
.py), and `GLUEProcessors` to [tasks/glue/processors.py](tasks/glue/processors.py), as follows.
```
GLUE_METRICS = {"mrpc": "acc"}
GLUE_PVPS = {"mrpc": MRPCPVP}
GLUE_PROCESSORS = {"mrpc": MRPCProcessor}
```

#### Step 5. Run your own Experiments
To run experiments on your new tasks with existing methods, based on the given scripts, you should simply replace the
 arguments
 `--dataset` and
 `--task` with your own ones (e.g., `--dataset glue` and `--task mrpc`).
 Besides, you may also need to adjust other dataset-related arguments such as `--max_seq_length` accordingly etc.



## Customizing your own Methods
To customize your own methods with FewNLU, you first need to create your own method repository in [methods](methods
). In sides the repository, you should define a model file, which implements the main parts of your new methods.

[To-be-updated]


## Citation
Please cite the paper if FewNLU is useful in your work:
```Bash
@misc{zheng2021fewnlu,
      title={FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding}, 
      author={Yanan Zheng and Jing Zhou and Yujie Qian and Ming Ding and Jian Li and Ruslan Salakhutdinov and Jie Tang and Sebastian Ruder and Zhilin Yang},
      year={2021},
      eprint={2109.12742},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement
Part of the code is based on [PET](https://github.com/timoschick/pet).
We appreciate all the contributors who made their code & dataset public, which greatly advanced few-shot learning as well as this FewNLU project. 
This repository will be continuously updated.
