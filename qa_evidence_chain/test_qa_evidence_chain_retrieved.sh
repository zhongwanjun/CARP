export RUN_ID=1
export BASIC_DIR=./ODQA
export MODEL_NAME=allenai/longformer-base-4096
export TOKENIZERS_PARALLELISM=false
export TRAIN_DATA_PATH=train_ranked_evidence_chain_for_qa_weighted.json
export DEV_DATA_PATH=dev_ranked_evidence_chain_for_qa_weighted.json
export MODEL_DIR=longformer_ecmask_weighted_1e-5_squadv1
export PREFIX=retrieved_blink_ecmask
python train_final_qa_ori.py \
    --do_predict \
    --model_type longformer \
    --evaluate_during_training \
    --data_dir ${BASIC_DIR}/data/qa_with_evidence_chain \
    --output_dir ${BASIC_DIR}/model/qa_model/${MODEL_DIR} \
    --train_file ${TRAIN_DATA_PATH} \
    --dev_file ${DEV_DATA_PATH} \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 10.0 \
    --max_seq_length 4096 \
    --doc_stride 3072 \
    --threads 8 \
    --topk_tbs 15 \
    --model_name_or_path ${MODEL_NAME} \
    --prefix ${PREFIX} \
    --save_cache \
    --overwrite_cache \
    --predict_file test_evidence_chain_weighted_scores.json \
    --pred_model_dir ${BASIC_DIR}/model/qa_model/longformer_ecmask_weighted_1e-5_squadv1/checkpoint-best \
    --predict_output_file ${BASIC_DIR}/submit_file/test_weighted_large_multineg_squadv1.json
