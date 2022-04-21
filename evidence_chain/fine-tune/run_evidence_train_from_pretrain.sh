export RUN_ID=5
export BASIC_PATH=./ODQA
export DATA_PATH=${BASIC_PATH}/evidence_chain_data/ground-truth-based/ground-truth-evidence-chain
export TRAIN_DATA_PATH=train_gt-ec-weighted.json
export DEV_DATA_PATH=dev_gt-ec-weighted.json
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large-new/checkpoint-best
export STEP=${step}
export PREFIX=southes-nonpretrain
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/ft_pretrained/roberta-base-${PREFIX}
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-pretrain-3neg-all-innerneg/checkpoint-${STEP}


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
--num_train_epochs 1 \

export TEST_DATA_PATH=dev_gttb_candidate_ec_weighted.json

python run_classifier.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name evidence_chain \
  --do_predict \
  --eval_all_checkpoints \
  --data_dir ${DATA_PATH} \
  --output_dir ${MODEL_PATH} \
  --max_seq_length 512 \
  --per_gpu_eval_batch_size 40 \
  --test_file ${DATA_PATH}/../candidate_chain/${TEST_DATA_PATH} \
  --pred_model_dir ${MODEL_PATH}/checkpoint-best \
  --test_result_dir ${DATA_PATH}/../scored_chain/pretrained/dev_roberta_base_scored_ec_addtab_weighted_${PREFIX}.json




