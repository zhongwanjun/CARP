export RUN_ID=1
export BASIC_DIR=./ODQA
export MODEL_NAME=allenai/longformer-base-4096
export TOKENIZERS_PARALLELISM=false
export TRAIN_DATA_PATH=${BASIC_DIR}/data/preprocessed_data/qa/train_intable_p1_t360.pkl
export DEV_DATA_PATH=${BASIC_DIR}/data/preprocessed_data/qa/dev_intable_p1_t360.pkl
export MODEL_DIR=qa_model_longformer_normalized_gtmodify_rankdoc_top15_newretr
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
    --do_train \
    --do_eval \
    --model_type longformer \
    --evaluate_during_training \
    --data_dir ${BASIC_DIR}/data/preprocessed_data/qa \
    --output_dir ${BASIC_DIR}/model/${MODEL_DIR} \
    --train_file train_preprocessed_normalized_gtmodify_newretr.json \
    --dev_file dev_preprocessed_normalized_gtmodify_newretr.json\
    --per_gpu_train_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 10.0 \
    --max_seq_length 4096 \
    --doc_stride 1024 \
    --threads 8 \
    --topk_tbs 15 \
    --model_name_or_path ${MODEL_NAME} \
    --repreprocess \
    --overwrite_cache \
    --prefix gtmodify_rankdoc_normalized_top15_newretr \
    2>&1 | tee ${BASIC_DIR}/qa_log/longformer-base-${MODEL_DIR}.log
