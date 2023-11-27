#!/bin/bash


echo ""
echo ""
echo ""

echo "RUNNING SCRIPT: $SLURM_JOB_NAME"


#########################
# conda 환경 활성화.
source ~/.bashrc
conda activate xlcost
# cuda 11.7 환경 구성.
ml purge
ml load cuda/11.7
#########################

# GPU 체크
nvidia-smi
nvcc -V

#####################################################################################################################






# model_type:         roberta                     codet5p                    plbart                 bart                  codet5                    roberta
# pretrained_model:   microsoft/unixcoder-base    Salesforce/codet5p-220m    uclanlp/plbart-base    facebook/bart-base    Salesforce/codet5-base    roberta-base

model_type=codet5p
pretrained_model=Salesforce/codet5p-220m

num_train_epochs=50

TRAIN_FILE_SRC=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/dataset/train.java
TRAIN_FILE_TGT=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/dataset/train.txt
VAL_FILE_SRC=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/dataset/valid.java
VAL_FILE_TGT=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/dataset/valid.txt
TEST_FILE_SRC=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/dataset/test.java
TEST_FILE_TGT=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/dataset/test.txt
SAVE_DIR=/home/ysnamgoong42/ws/BANKWAREGLOBAL/#WORKSPACE/4_2023-11-24_CodeBase/saved_models/${pretrained_model}

#####################################################################################################################





function train () {

python run.py \
    --do_train \
    --do_eval \
    --model_type $model_type \
    --config_name $pretrained_model \
    --tokenizer_name $pretrained_model \
    --model_name_or_path $pretrained_model \
    --train_filename $TRAIN_FILE_SRC,$TRAIN_FILE_TGT \
    --dev_filename $VAL_FILE_SRC,$VAL_FILE_TGT \
    --output_dir $SAVE_DIR \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_train_epochs $num_train_epochs \
    --train_steps 5000 \
    --eval_steps 2500 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --beam_size 5 \
    --learning_rate 5e-5 \

}


function evaluate () {

MODEL_PATH=${SAVE_DIR}/checkpoint-best-ppl/pytorch_model.bin;
RESULT_FILE=${SAVE_DIR}/result.txt;

python run.py \
    --do_test \
    --model_type $model_type \
    --model_name_or_path $pretrained_model \
    --config_name $pretrained_model \
    --tokenizer_name $pretrained_model  \
    --load_model_path $MODEL_PATH \
    --test_filename $TEST_FILE_SRC,$TEST_FILE_TGT \
    --output_dir $SAVE_DIR \
    --max_source_length 512 \
    --max_target_length 128 \
    --beam_size 5 \
    --eval_batch_size 16 \

python evaluator.py \
    --references $TEST_FILE_TGT \
    --predictions ${SAVE_DIR}/test_0.output \
    2>&1 | tee $RESULT_FILE;

}

train;

evaluate;

