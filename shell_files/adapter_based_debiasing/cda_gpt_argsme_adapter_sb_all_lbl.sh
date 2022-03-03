#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file ../data/cda/cda_argsme_all_targets_sb_train.txt \
    --validation_file ../data/cda/cda_argsme_all_targets_sb_test.txt \
    --evaluation_strategy "epoch" \
    --num_train_epochs 5 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --fp16 True \
    --output_dir output_logs/gpt2_argsme_adapter_cda_sb_all_lbl \
    --train_adapter \
    --adapter_config "pfeiffer+inv" \
    --block_size 512 \
    --gradient_accumulation_steps 8 \
    --load_best_model_at_end True \
    --dataloader_pin_memory True \
    --cache_dir cache_dir_gpt_cda_sb_argsme_all_lbl \
    --line_by_line True \
    --overwrite_cache