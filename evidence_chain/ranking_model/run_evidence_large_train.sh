export RUN_ID=5
export BASIC_PATH=./ODQA
export DATA_PATH=${BASIC_PATH}/data/preprocessed_data/evidence_chain/ground-truth-based
export TRAIN_DATA_PATH=train_preprocessed_normalized_gtmodify_evichain_addnoun.json
export DEV_DATA_PATH=dev_preprocessed_normalized_gtmodify_evichain_addnoun.json
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-ranking-pretrain
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large-pretrain-ec-ranker/checkpoint-best
python run_classifier.py \
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

