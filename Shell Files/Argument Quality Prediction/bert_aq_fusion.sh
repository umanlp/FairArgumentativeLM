#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

for EPOCHS in 1.0 2.0 3.0 4.0 5.0; do
    for LEARNING_RATE in 2e-4 1e-4 3e-4; do
        DATA_DIR="../data/argument_quality/"
        SEED=100
        BATCH_SIZE=32
        TASK_ADAPTER_DIR='output_logs/argument_quality/bert_seed_ibm_overall_5.0_3e-4_50/ibm'
        ADAPTER_DIR1='output_logs/bert_argsme_adapter_cda_rb_all_subset_lbl/mlm'
        ADAPTER_DIR2='output_logs/bert_ft_cmv/mlm'
        TASK_NAME='ibm'
        TARGET_LABEL='overall'
        OUTPUT_DIR="output_logs/argument_quality/bert_fusion_rb_${TASK_NAME}_${TARGET_LABEL}_${EPOCHS}_${LEARNING_RATE}_${SEED}/"
        CACHE_DIR=cache_arg_qual

        python argument_quality_prediction_fusion.py \
            --model_name_or_path bert-base-uncased \
            --train_file '../data/argument_quality/ibm_rank_train.csv' \
            --validation_file '../data/argument_quality/ibm_rank_dev.csv' \
            --test_file '../data/argument_quality/ibm_rank_test.csv' \
            --train_adapter \
            --load_lang_adapter ${ADAPTER_DIR1} \
            --lang_adapter_config pfeiffer+inv \
            --adapter_config pfeiffer \
            --load_second_lang_adapter ${ADAPTER_DIR2} \
            --second_lang_adapter_config pfeiffer+inv \
            --load_task_adapter ${TASK_ADAPTER_DIR} \
            --task_adapter_config pfeiffer \
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