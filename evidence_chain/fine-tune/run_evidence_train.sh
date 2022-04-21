export RUN_ID=5
export BASIC_PATH=./ODQA
export DATA_PATH=${BASIC_PATH}/data/evidence_chain_data/ground-truth-based/ground-truth-evidence-chain
export TRAIN_DATA_PATH=train_gt-ec.jsonl
export DEV_DATA_PATH=dev_gt-ec.jsonl
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large-new/checkpoint-best
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run_classifier.py \
--model_type roberta \
--tokenizer_name roberta-large \
--config_name roberta-large \
--model_name_or_path roberta-large \
--task_name evidence_chain \
--overwrite_cache \
--do_train \
--do_eval \
--eval_all_checkpoints \
--data_dir ${DATA_PATH} \
--output_dir ${MODEL_PATH} \
--train_file ${TRAIN_DATA_PATH} \
--dev_file ${DEV_DATA_PATH} \
--max_seq_length 512 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 1e-5 \
--num_train_epochs 10
