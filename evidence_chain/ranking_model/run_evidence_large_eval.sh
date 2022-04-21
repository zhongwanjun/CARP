export RUN_ID=5
export BASIC_PATH=./ODQA
export DATA_PATH=${BASIC_PATH}/data/preprocessed_data/evidence_chain/ground-truth-based
export TRAIN_DATA_PATH=train_preprocessed_normalized_gtmodify_evichain_nx.json
export DEV_DATA_PATH=dev_preprocessed_normalized_gtmodify_evichain_nx.json
export TEST_DATA_PATH=train_preprocessed_normalized_gtmodify_candidate_evichain_addnoun.json
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-ecpretrain
export TEST_DATA_PATH=dev_preprocessed_normalized_gtmodify_candidate_evichain_addnoun.json
#export TEST_DATA_PATH=new_chain_blink_dev_evidence_chain_ranking.json
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
	--per_gpu_eval_batch_size 40 \
	--overwrite_cache \
	--test_file ${DATA_PATH}/${TEST_DATA_PATH} \
	--pred_model_dir ${MODEL_PATH}/checkpoint-best \
	--test_result_dir ${DATA_PATH}/dev_ecpretrain_ranker_scores.json