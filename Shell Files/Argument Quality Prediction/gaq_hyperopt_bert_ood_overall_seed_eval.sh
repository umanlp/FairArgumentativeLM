#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

for EPOCHS in 5.0; do
    for LEARNING_RATE in 3e-4; do 
        for DIMENSION in 'qa' 'rev'; do
            for SEED in 45 4 93 65 18 12 53 44 71 7 96 94 76 33 50 79 17 38 25 3 5 73 91 49 86 57 35 80 52 62 54 42 34 98 16 78 20 82 67 39 9 88 27 69 77 21 41 74 87 60; do
                printf "$DIMENSION\n"
                printf "$SEED\n"
                DATA_DIR="../data/argument_quality/"
                BATCH_SIZE=32
                TASK_NAME='gaq'
                TARGET_LABEL='overall'
                OUTPUT_DIR="output_logs/argument_quality/bert_od_${DIMENSION}_seed_${TASK_NAME}_${TARGET_LABEL}_${EPOCHS}_${LEARNING_RATE}_${SEED}/"
                CACHE_DIR=cache_arg_qual_gaq

                python adapt_glue_file.py \
                    --model_name_or_path bert-base-uncased \
                    --train_file '../data/argument_quality/gaq_ood_train_target_overall_mean.csv' \
                    --validation_file '../data/argument_quality/gaq_ood_dev_target_overall_mean.csv' \
                    --test_file "../data/argument_quality/gaq_ood_test_${DIMENSION}_target_overall_mean.csv" \
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
    done;
done;