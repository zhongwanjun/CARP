export RUN_ID=1
export BASIC_DIR=/wanjun/ODQA
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
cp train_qa_wajun_shuffle.sh ${BASIC_DIR}/model/${MODEL_DIR}/
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
#     --do_train \
#     --do_eval \
#     --model_type roberta \
#     --evaluate_during_training \
#     --data_dir ${BASIC_DIR}/data/preprocessed_data/qa_evidence_chain \
#     --output_dir ${BASIC_DIR}/model/qa_model_roberta_large \
#     --train_file train_preprocessed.json \
#     --dev_file dev_preprocessed.json\
#     --per_gpu_train_batch_size 1 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 10.0 \
#     --max_seq_length 512 \
#     --doc_stride 128 \
#     --threads 8 \
#     --topk_tbs 4 \
#     --retrieval_results_file_dev ${BASIC_DIR}/data/preprocessed_data/qa_evidence_chain/roberta_meta_training/dev_output_k100.json \
#     --retrieval_results_file_train ${BASIC_DIR}/data/preprocessed_data/qa_evidence_chain/roberta_meta_training/train_output_k100.json \
#     --model_name_or_path roberta-large \
#     --repreprocess \
#     2>&1 | tee ${BASIC_DIR}/qa_log/roberta-large.log
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
#    --do_train \
#    --do_eval \
#    --do_lower_case  \
#    --model_type bert \
#    --evaluate_during_training \
#    --data_dir ../preprocessed_data/qa_evidence_chain \
#    --output_dir ../models/qa_model/qa_bert-base-cased \
#    --train_file train_preprocessed.json \
#    --dev_file dev_preprocessed.json\
#    --per_gpu_train_batch_size 16  \
# 	--per_gpu_eval_batch_size 16  \
#    --learning_rate 3e-5  \
#    --num_train_epochs 4.0  \
#    --max_seq_length 512  \
#    --doc_stride 128  \
#    --threads 8 \
#    --model_name_or_path bert-base-cased \
#    2>&1 |tee ./run_logs/qa_bert-base-cased.log
