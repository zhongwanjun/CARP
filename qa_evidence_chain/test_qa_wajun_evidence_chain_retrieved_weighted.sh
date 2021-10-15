export RUN_ID=1
#export BASIC_DIR=/wanjun/ODQA
export BASIC_DIR=/home/t-wzhong/v-wanzho/ODQA
export MODEL_NAME=allenai/longformer-base-4096
export TOKENIZERS_PARALLELISM=false
export TRAIN_DATA_PATH=train_ranked_evidence_chain_for_qa_weighted.json
export DEV_DATA_PATH=dev_ranked_evidence_chain_for_qa_weighted.json
#export MODEL_DIR=qa_model_
export MODEL_DIR=longformer_ecmask_weighted_1e-5_squadv1
export PREFIX=retrieved_blink_ecmask
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa_ori.py \
    --do_predict \
    --model_type longformer \
    --evaluate_during_training \
    --data_dir ${BASIC_DIR}/data/preprocessed_data/evidence_chain/retrieval-based/data4qa/weighted_large \
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
    --predict_output_file /home/t-wzhong/v-wanzho/ODQA/submit_file/test_weighted_large_multineg_squadv1.json
    2>&1 | tee ${BASIC_DIR}/qa_log/longformer-base-${MODEL_DIR}.log
cp train_qa_wajun_evidence_chain_retrieved.sh ${BASIC_DIR}/model/qa_model/${MODEL_DIR}/
#
#export RUN_ID=1 
#export BASIC_DIR=/wanjun/ODQA
#export MODEL_NAME=allenai/longformer-base-4096
#export TOKENIZERS_PARALLELISM=false
#export TRAIN_DATA_PATH=${BASIC_DIR}/data/preprocessed_data/qa/train_intable_p1_t360.pkl
#export DEV_DATA_PATH=${BASIC_DIR}/data/preprocessed_data/qa/dev_intable_p1_t360.pkl
#CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
#    --do_train \
#    --do_eval \
#    --model_type longformer \
#    --evaluate_during_training \
#    --data_dir ${BASIC_DIR}/data/preprocessed_data/qa \
#    --output_dir ${BASIC_DIR}/model/qa_model_longformer_meta_v2_top10 \
#    --train_file train_preprocessed_v2.json \
#    --dev_file dev_preprocessed_v2.json\
#    --per_gpu_train_batch_size 1 \
#    --learning_rate 3e-5 \
#    --num_train_epochs 10.0 \
#    --max_seq_length 4096 \
#    --doc_stride 512 \
#    --threads 8 \
#    --topk_tbs 10 \
#    --model_name_or_path ${MODEL_NAME} \
#    --overwrite_cache \ 
#    2>&1 | tee ${BASIC_DIR}/qa_log/longformer-base-meta-v3-top10.log
#CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
#    --do_train \
#    --do_eval \
#    --model_type longformer \
#    --evaluate_during_training \
#    --data_dir ${BASIC_DIR}/data/preprocessed_data/qa \
#    --output_dir ${BASIC_DIR}/model/qa_model_longformer_meta_v2_top15 \
#    --train_file train_preprocessed_v2.json \
#    --dev_file dev_preprocessed_v2.json\
#    --per_gpu_train_batch_size 1 \
#    --learning_rate 3e-5 \
#    --num_train_epochs 10.0 \
#    --max_seq_length 4096 \
#    --doc_stride 1024 \
#    --threads 8 \
#    --topk_tbs 15 \
#    --model_name_or_path ${MODEL_NAME} \
#    --overwrite_cache \
#    2>&1 | tee ${BASIC_DIR}/qa_log/longformer-base-meta-v3-top15.log
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
#     --do_train \
#     --do_eval \
#     --model_type roberta \
#     --evaluate_during_training \
#     --data_dir ${BASIC_DIR}/data/preprocessed_data/qa \
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
#     --retrieval_results_file_dev ${BASIC_DIR}/data/preprocessed_data/qa/roberta_meta_training/dev_output_k100.json \
#     --retrieval_results_file_train ${BASIC_DIR}/data/preprocessed_data/qa/roberta_meta_training/train_output_k100.json \
#     --model_name_or_path roberta-large \
#     --repreprocess \
#     2>&1 | tee ${BASIC_DIR}/qa_log/roberta-large.log
#CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
#    --do_train \
#    --do_eval \
#    --do_lower_case  \
#    --model_type bert \
#    --evaluate_during_training \
#    --data_dir ../preprocessed_data/qa \
#    --output_dir ../models/qa_model/qa_bert-base-cased \
#    --train_file train_preprocessed.json \
#    --dev_file dev_preprocessed.json\
#    --per_gpu_train_batch_size 16  \
#	--per_gpu_eval_batch_size 16  \
#    --learning_rate 3e-5  \
#    --num_train_epochs 4.0  \
#    --max_seq_length 512  \
#    --doc_stride 128  \
#    --threads 8 \
#    --model_name_or_path bert-base-cased \
#    2>&1 |tee ./run_logs/qa_bert-base-cased.log
