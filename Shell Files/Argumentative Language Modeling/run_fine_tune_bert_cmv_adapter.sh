#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python fine_tune_mlm_adapter.py 
--model_name_or_path bert-base-uncased 
--train_file ../data/cmv/ft_training.txt 
--validation_file ../data/cmv/ft_test.txt 
--evaluation_strategy "epoch" 
--num_train_epochs 10 
--do_train 
--do_eval 
--per_device_train_batch_size 8 
--per_device_eval_batch_size 8 
--learning_rate 1e-4 
--weight_decay 0.01 
--fp16 True 
--output_dir output_logs/bert_cmv_adapter 
--train_adapter 
--adapter_config "pfeiffer+inv" 
--gradient_accumulation_steps 4 
--load_best_model_at_end True 
--dataloader_pin_memory True 
--cache_dir cache_dir_bert_cmv