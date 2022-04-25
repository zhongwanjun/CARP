export RUN_ID=1
export BASIC_DIR=./ODQA
export MODEL_NAME=allenai/longformer-base-4096
export TOKENIZERS_PARALLELISM=false
export TRAIN_DATA_PATH=train_ranked_evidence_chain_for_qa_weighted.json
export DEV_DATA_PATH=dev_ranked_evidence_chain_for_qa_weighted.json
export MODEL_DIR=${MODEL_CHECKPOINT}
export PREFIX=retrieved_blink_ecmask
python train_final_qa_ori.py \
    --do_eval \
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
