task_name=$1
device=$2
model_type=$3

few_shot_setting="dev32_split"
dataset_name="superglue"
method="ptuning"
arch_method="default"
data_dir=$YOUR_DATA_DIR
save_dir=$YOUR_SAVE_DIR/${few_shot_setting}/${model_type}_${task_name}_${arch_method}_${method}_model


if [ $model_type = "albert" ]; then
  model_name_or_path="albert-xxlarge-v2"
  TRAIN_BATCH_SIZE_CANDIDATES="8"
  LR_CANDIDATES="1e-5 2e-5"

elif [ $model_type = "deberta" ]; then
  model_name_or_path="microsoft/deberta-v2-xxlarge"
  TRAIN_BATCH_SIZE_CANDIDATES="2"
  LR_CANDIDATES="1e-5 5e-6"
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
TOTAL_BATCH_SIZE_CANDIDATES="16"
EVAL_BATCH_SIZE=32
DATA_ROOT=$data_dir
TASK=$task_name
PATTERN_IDS="1"


if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTEs

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128
  EVAL_BATCH_SIZE=1

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96
  EVAL_BATCH_SIZE=1

elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  TRAIN_BATCH_SIZE=1

elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  TRAIN_BATCH_SIZE=1

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi


MAX_STEPS="250 500"
WARMUP_RATIO="0.0 0.1"
SAMPLER_SEED="42"
SEED="42"
every_eval_ratios="0.02 0.04"
HEAD_TYPES="lstm mlp"
cv_k="4"



for MAX_STEP in $MAX_STEPS
do
  for TOTAL_TRAIN_BATCH in $TOTAL_BATCH_SIZE_CANDIDATES
  do
    for TRAIN_BATCH_SIZE in $TRAIN_BATCH_SIZE_CANDIDATES
    do
      for warmup_ratio in $WARMUP_RATIO
      do
        for every_eval_ratio in $every_eval_ratios
        do
          for LR in $LR_CANDIDATES
          do
            for prompt_encoder_head_type in $HEAD_TYPES
            do

            if [ $LR = '5e-6' ]; then
              EMB_LR=1e-5
            elif [ $LR = '1e-5' ]; then
              EMB_LR=1e-4
            elif [ $LR = '2e-5' ]; then
              EMB_LR=1e-4
            else
              exit 1
            fi

            ACCU=$((${TOTAL_TRAIN_BATCH}/${TRAIN_BATCH_SIZE}))
            HYPER_PARAMS=${SEQ_LENGTH}_${MAX_STEP}_${TOTAL_TRAIN_BATCH}_${TRAIN_BATCH_SIZE}_${LR}_${PATTERN}_${every_eval_ratio}_${warmup_ratio}_${prompt_encoder_head_type}_${cv_k}
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
              --warmup_step_ratio $warmup_ratio \
              --learning_rate $LR \
              --repetitions 1 \
              --embedding_learning_rate $EMB_LR \
              --use_continuous_prompt \
              --prompt_encoder_head_type $prompt_encoder_head_type \
              --few_shot_setting $few_shot_setting \
              --every_eval_ratio $every_eval_ratio \
              --cv_k $cv_k \
              --fix_deberta #>myout_${few_shot_setting}_${method}_${task_name}.file 2>&1 &
              wait
            done
          done
        done
      done
    done
  done
done


# for ALBERT:
# nohup bash search_ptuning_multisplit.sh boolq 0 albert >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh rte 1 albert >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh wic 2 albert >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh cb 3 albert >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh multirc 4 albert >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh wsc 5 albert >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh copa 6 albert >myout.file 2>&1 &

# for DeBERTa:
# nohup bash search_ptuning_multisplit.sh boolq 0 deberta >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh rte 1 deberta >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh wic 2 deberta >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh cb 3 deberta >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh multirc 4 deberta >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh wsc 5 deberta >myout.file 2>&1 &
# nohup bash search_ptuning_multisplit.sh copa 6 deberta >myout.file 2>&1 &


