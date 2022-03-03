#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

for EPOCHS in 5.0; do
    for LEARNING_RATE in 3e-4; do
        for SEED in 45 4 93 65 18 12 53 44 71 7 96 94 76 33 50 79 17 38 25 3 5 73 91 49 86 57 35 80 52 62 54 42 34 98 16 78 20 82 67 39 9 88 27 69 77 21 41 74 87 60; do
            printf "$SEED\n"
            DATA_DIR="../data/argument_quality/"
            BATCH_SIZE=32
            TASK_NAME='gaq'
            TARGET_LABEL='overall'
            DIMENSION='deb'
            printf "$DIMENSION\n"
            OUTPUT_DIR="output_logs/argument_quality/bert_id_ft_argsme_${DIMENSION}_seed_${TASK_NAME}_${TARGET_LABEL}_${EPOCHS}_${LEARNING_RATE}_${SEED}/"
            CACHE_DIR=cache_arg_qual_gaq

            python adapt_glue_file.py \
                --model_name_or_path bert-base-uncased \
                --train_file "../data/argument_quality/gaq_id_train_${DIMENSION}.csv" \
                --validation_file "../data/argument_quality/gaq_id_dev_${DIMENSION}.csv" \
                --test_file "../data/argument_quality/gaq_id_test_${DIMENSION}.csv" \
                --train_adapter \
                --load_lang_adapter 'output_logs/bert_ft_argsme/mlm'\
                --lang_adapter_config pfeiffer+inv \
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

for EPOCHS in 5.0; do
    for LEARNING_RATE in 3e-4; do
        for SEED in 45 4 93 65 18 12 53 44 71 7 96 94 76 33 50 79 17 38 25 3 5 73 91 49 86 57 35 80 52 62 54 42 34 98 16 78 20 82 67 39 9 88 27 69 77 21 41 74 87 60; do
            printf "$SEED\n"
            DATA_DIR="../data/argument_quality/"
            BATCH_SIZE=32
            TASK_NAME='gaq'
            TARGET_LABEL='overall'
            DIMENSION='rev'
            printf "$DIMENSION\n"
            OUTPUT_DIR="output_logs/argument_quality/bert_id_ft_argsme_${DIMENSION}_seed_${TASK_NAME}_${TARGET_LABEL}_${EPOCHS}_${LEARNING_RATE}_${SEED}/"
            CACHE_DIR=cache_arg_qual_gaq

            python adapt_glue_file.py \
                --model_name_or_path bert-base-uncased \
                --train_file "../data/argument_quality/gaq_id_train_${DIMENSION}.csv" \
                --validation_file "../data/argument_quality/gaq_id_dev_${DIMENSION}.csv" \
                --test_file "../data/argument_quality/gaq_id_test_${DIMENSION}.csv" \
                --train_adapter \
                --load_lang_adapter 'output_logs/bert_ft_argsme/mlm'\
                --lang_adapter_config pfeiffer+inv \
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

for EPOCHS in 4.0; do
    for LEARNING_RATE in 3e-4; do
        for SEED in 45 4 93 65 18 12 53 44 71 7 96 94 76 33 50 79 17 38 25 3 5 73 91 49 86 57 35 80 52 62 54 42 34 98 16 78 20 82 67 39 9 88 27 69 77 21 41 74 87 60; do
            printf "$SEED\n"
            DATA_DIR="../data/argument_quality/"
            BATCH_SIZE=32
            TASK_NAME='gaq'
            TARGET_LABEL='overall'
            DIMENSION='qa'
            printf "$DIMENSION\n"
            OUTPUT_DIR="output_logs/argument_quality/bert_id_ft_argsme_${DIMENSION}_seed_${TASK_NAME}_${TARGET_LABEL}_${EPOCHS}_${LEARNING_RATE}_${SEED}/"
            CACHE_DIR=cache_arg_qual_gaq

            python adapt_glue_file.py \
                --model_name_or_path bert-base-uncased \
                --train_file "../data/argument_quality/gaq_id_train_${DIMENSION}.csv" \
                --validation_file "../data/argument_quality/gaq_id_dev_${DIMENSION}.csv" \
                --test_file "../data/argument_quality/gaq_id_test_${DIMENSION}.csv" \
                --train_adapter \
                --load_lang_adapter 'output_logs/bert_ft_argsme/mlm'\
                --lang_adapter_config pfeiffer+inv \
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