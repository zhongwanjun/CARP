export RUN_ID=5
export BASIC_PATH=./ODQA
export DATA_PATH=${BASIC_PATH}/data/evidence_chain_data/bart_output_for_pretraining/add_negatives/pre-training/
export TRAIN_DATA_PATH=evidence_pretrain_train_shortest_esnegs.jsonl
export DEV_DATA_PATH=evidence_pretrain_dev_shortest_esnegs.jsonl
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-pretrain-1neg-weighted-esneg
python run_classifier.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name evidence_chain \
--do_predict \
--eval_all_checkpoints \
--data_dir ${DATA_PATH} \
--output_dir ${MODEL_PATH} \
--train_file ${TRAIN_DATA_PATH} \
--dev_file ${DEV_DATA_PATH} \
--max_seq_length 512 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 64 \
--learning_rate 3e-5 \
--evaluate_during_training \
--overwrite_cache \
--save_steps 6000 \
--num_train_epochs 3 \
--pred_model_dir ${MODEL_PATH}/checkpoint-best \
--test_file ${DEV_DATA_PATH} \
--test_result_dir ${MODEL_PATH}/eval_results.txt \
