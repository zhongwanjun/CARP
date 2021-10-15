export RUN_ID=5
#export BASIC_PATH=/wanjun/ODQA
export BASIC_PATH=/home/t-wzhong/v-wanzho/ODQA
export DATA_PATH=${BASIC_PATH}/data/preprocessed_data/evidence_chain/ground-truth-based
export TRAIN_DATA_PATH=train_preprocessed_normalized_gtmodify_evichain_addnoun.json
export DEV_DATA_PATH=dev_preprocessed_normalized_gtmodify_evichain_addnoun.json
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-ranking-pretrain
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large-pretrain-ec-ranker/checkpoint-best
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run_classifier.py \
        --model_type roberta \
        --tokenizer_name roberta-base \
        --config_name roberta-base \
        --model_name_or_path roberta-base \
        --task_name evidence_chain \
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
        --num_train_epochs 10 \
        --load_save_pretrain \
        --pretrain_model_dir ${PRETRAIN_MODEL_PATH} \

#--test_file /home/dutang/FEVER/arranged_data/bert_data/Evidence/eval_file/train_all_sentence_evidence_title_all.tsv \
#--pred_model_dir /home/dutang/FEVER/arranged_models/xlnet_torch_models/evidence_large_title/checkpoint-best \
#--test_result_dir /home/dutang/FEVER/arranged_result/xlnet/evidence/train_evidence_xlnet_large_title_score.tsv
