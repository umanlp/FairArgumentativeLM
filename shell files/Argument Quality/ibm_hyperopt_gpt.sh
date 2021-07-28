#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

for EPOCHS in 1.0 2.0 3.0 4.0 5.0; do
    for LEARNING_RATE in 2e-4 1e-4 3e-4; do
        DATA_DIR="../data/argument_quality/"
        SEED=100
        BATCH_SIZE=32
        TASK_NAME='ibm'
        TARGET_LABEL='overall'
        OUTPUT_DIR="output_logs/argument_quality/gpt_${TASK_NAME}_${TARGET_LABEL}_${EPOCHS}_${LEARNING_RATE}_${SEED}/"
        CACHE_DIR=cache_arg_qual

        python adapt_glue_file.py \
            --model_name_or_path gpt2 \
            --train_file '../data/argument_quality/ibm_rank_train_gpt.csv' \
            --validation_file '../data/argument_quality/ibm_rank_dev_gpt.csv' \
            --test_file '../data/argument_quality/ibm_rank_test_gpt.csv' \
            --train_adapter \
            --adapter_config pfeiffer \
            --task_name ${TASK_NAME} \
            --do_train \
            --do_eval \
            --do_predict \
            --max_seq_length 128 \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --num_train_epochs ${EPOCHS} \
            --output_dir ${OUTPUT_DIR} \
            --seed ${SEED} \
            --cache_dir ${CACHE_DIR} \
            --overwrite_cache \
            --overwrite_output_dir \
            --save_steps -1 \


    done;
done;