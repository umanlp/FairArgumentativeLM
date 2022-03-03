#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file ../data/cda/cda_wikipedia_df_rb_train.txt \
    --validation_file ../data/cda/cda_wikipedia_df_rb_test.txt \
    --evaluation_strategy "epoch" \
    --num_train_epochs 5 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --fp16 True \
    --output_dir output_logs/bert_wiki_adapter_cda_rb_ta_lbl \
    --train_adapter \
    --adapter_config "pfeiffer+inv" \
    --gradient_accumulation_steps 8 \
    --load_best_model_at_end True \
    --dataloader_pin_memory True \
    --cache_dir cache_dir_bert_cda_rb_wiki_ta_lbl \
    --line_by_line True
