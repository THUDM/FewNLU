task_name=$1
device=$2
LR=$3
MAX_STEP=$4
prompt_encoder_head_type=$5
every_eval_ratio=$6
arch_method=$7
model_type=$8

PATTERN_IDS=1

few_shot_setting="dev32_split"
dataset_name="superglue"
method="ptuning"

data_dir=$YOUR_DATA_DIR
save_dir=$YOUR_SAVE_DIR/${few_shot_setting}/${model_type}_${task_name}_${arch_method}_cross_${method}_model

unlabeled_examples=500
ipet_scale_factor=3
ipet_logits_percentage=0.5

if [ $model_type = "albert" ]; then
  model_name_or_path="albert-xxlarge-v2"
  TRAIN_BATCH_SIZE=8

elif [ $model_type = "deberta" ]; then
  model_name_or_path="microsoft/deberta-v2-xxlarge"
  TRAIN_BATCH_SIZE=2
fi


echo Running with the following parameters:
echo ------------------------------------
echo DATASET_NAME           = "$dataset_name"
echo TASK_NAME              = "$task_name"
echo METHOD                 = "$method"
echo DEVICE                 = "$device"
echo MODEL_TYPE             = "$model_type"
echo MODEL_NAME_OR_PATH     = "$model_name_or_path"
echo DATA_ROOT              = "$data_dir"
echo SAVE_DIR               = "$save_dir"
echo ------------------------------------


SEQ_LENGTH=256
TOTAL_TRAIN_BATCH=16
EVAL_BATCH_SIZE=32
DATA_ROOT=$data_dir
TASK=$task_name


if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTE

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128
  EVAL_BATCH_SIZE=1
  max_num_lbl_tok=20

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96
  EVAL_BATCH_SIZE=1
  max_num_lbl_tok=20


elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  TRAIN_BATCH_SIZE=1


elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi


WARMUP_RATIO="0.0"
SAMPLER_SEED="42 43 44"
SEED="42"
cv_k="4"


if [ $LR == '5e-6' ]; then
  EMB_LR=1e-5
elif [ $LR == '1e-5' ]; then
  EMB_LR=1e-4
elif [ $LR == '2e-5' ]; then
  EMB_LR=1e-4
else
  exit 1
fi

ACCU=$((${TOTAL_TRAIN_BATCH}/${TRAIN_BATCH_SIZE}))
HYPER_PARAMS=${SEQ_LENGTH}_${MAX_STEP}_16_2_${LR}_1_${every_eval_ratio}_${warmup_ratio}_${prompt_encoder_head_type}_${unlabeled_examples}_${PATTERN_IDS}_${ipet_scale_factor}_${ipet_logits_percentage}
OUTPUT_DIR=$save_dir/${HYPER_PARAMS}
CUDA_VISIBLE_DEVICES=$device python3 cli.py \
  --method $method \
  --arch_method $arch_method \
  --data_dir $DATA_DIR \
  --pattern_ids $PATTERN_IDS \
  --model_type $model_type \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name \
  --task_name $task_name \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --do_train \
  --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
  --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $ACCU \
  --max_seq_length $SEQ_LENGTH \
  --max_steps $MAX_STEP \
  --sampler_seed $SAMPLER_SEED \
  --seed $SEED \
  --warmup_step_ratio $WARMUP_RATIO \
  --learning_rate $LR \
  --repetitions 1 \
  --embedding_learning_rate $EMB_LR \
  --use_continuous_prompt \
  --prompt_encoder_head_type $prompt_encoder_head_type \
  --few_shot_setting $few_shot_setting \
  --every_eval_ratio $every_eval_ratio \
  --cv_k $cv_k \
  --fix_deberta \
  --ipet_logits_percentage $ipet_logits_percentage \
  --ipet_scale_factor $ipet_scale_factor \
  --use_brother_fold_logits \
  --unlabeled_examples $unlabeled_examples


# bash search_semi_multisplit_ptuning_cross.sh boolq 0 5e-6 500 mlp 0.02 ipet deberta
# bash search_semi_multisplit_ptuning_cross.sh rte 1 5e-6 250 lstm 0.02 ipet deberta 
# bash search_semi_multisplit_ptuning_cross.sh wic 0 5e-6 500 lstm 0.02 ipet deberta 
# bash search_semi_multisplit_ptuning_cross.sh cb 0 1e-5 250 lstm 0.04 ipet deberta 
# bash search_semi_multisplit_ptuning_cross.sh multirc 0 1e-5 500 lstm 0.02 ipet deberta 
# bash search_semi_multisplit_ptuning_cross.sh wsc 0 1e-5 250 mlp 0.04 ipet deberta
# bash search_semi_multisplit_ptuning_cross.sh copa 7 1e-5 500 mlp 0.04 ipet deberta


# bash search_semi_multisplit_ptuning_cross.sh boolq 0 5e-6 500 mlp 0.02 noisy_student deberta
# bash search_semi_multisplit_ptuning_cross.sh rte 0 5e-6 250 lstm 0.02 noisy_student deberta 
# bash search_semi_multisplit_ptuning_cross.sh wic 0 5e-6 500 lstm 0.02 noisy_student deberta 
# bash search_semi_multisplit_ptuning_cross.sh cb 0 1e-5 250 lstm 0.04 noisy_student deberta 
# bash search_semi_multisplit_ptuning_cross.sh multirc 0 1e-5 500 lstm 0.02 noisy_student deberta 
# bash search_semi_multisplit_ptuning_cross.sh wsc 0 1e-5 250 mlp 0.04 noisy_student deberta
# bash search_semi_multisplit_ptuning_cross.sh copa 0 1e-5 500 mlp 0.04 noisy_student deberta


