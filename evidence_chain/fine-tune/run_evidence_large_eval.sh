export RUN_ID=5
export BASIC_PATH=./ODQA
#export DATA_PATH=${BASIC_PATH}/data/preprocessed_data/evidence_chain/ground-truth-based/candidate_chain
export DATA_PATH=./ODQA/data/evidence_chain_data/retrieval-based/candidate_chain
export TRAIN_DATA_PATH=train_preprocessed_normalized_gtmodify_evichain_nx.json
export DEV_DATA_PATH=dev_preprocessed_normalized_gtmodify_evichain_nx.json
export TEST_DATA_PATH=dev_gttb_candidate_ec.jsonl
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large-weighted-multineg
for part in 2 3
do
  export TEST_DATA_PATH=train_evidence_chain_weighted_ranking_${part}.json
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run_classifier.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name evidence_chain \
    --do_predict \
    --eval_all_checkpoints \
    --data_dir ${DATA_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_file ${TRAIN_DATA_PATH} \
    --dev_file ${DEV_DATA_PATH} \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 20 \
    --overwrite_cache \
    --test_file ${DATA_PATH}/${TEST_DATA_PATH} \
    --pred_model_dir ${MODEL_PATH}/checkpoint-best \
    --test_result_dir ${DATA_PATH}/../scored_chains/weighted-large-5neg/train_evidence_chain_weighted_scores_${part}.json
    sleep 30
done
