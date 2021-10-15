export RUN_ID=5
#export BASIC_PATH=/wanjun/ODQA
export BASIC_PATH=/home/t-wzhong/v-wanzho/ODQA
export DATA_PATH=${BASIC_PATH}/data/preprocessed_data/evidence_chain/ground-truth-based/ground-truth-evidence-chain
export TRAIN_DATA_PATH=train_gt-ec-weighted.json
export DEV_DATA_PATH=dev_gt-ec-weighted.json
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-large-new/checkpoint-best
export STEP=${step}
export PREFIX=southes-nonpretrain
export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/ft_pretrained/roberta-base-${PREFIX}
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-pretrain-3neg-all-innerneg/checkpoint-${STEP}


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
--num_train_epochs 1 \
2>&1 | tee ~/v-wanzho/ODQA/code/advance-qa/evidence_chain/pretrain_logs/ft-pretrain-${PREFIX}.log

export TEST_DATA_PATH=dev_gttb_candidate_ec_weighted.json


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run_classifier.py \
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
#python ~/v-wanzho/ODQA/code/advance-qa_evidence_chain/evidence_chain/extraction/evaluate_ranked_evidence_chain.py pretrained/dev_roberta_base_scored_ec_addtab_weighted_${PREFIX}.json 2>&1 | tee ~/v-wanzho/ODQA/code/advance-qa_evidence_chain/evidence_chain/pretrain_logs/evaluate_performance_${PREFIX}.log
export PREFIX=all-3innerneg-all-weighted
for step in best 6000 12000 18000 24000 30000
do
    export STEP=${step}
    export MODEL_PATH=${BASIC_PATH}/model/evidence_chain/ft_pretrained/roberta-base-pretrain-${PREFIX}-${STEP}
    export PRETRAIN_MODEL_PATH=${BASIC_PATH}/model/evidence_chain/roberta-base-pretrain-3neg-weighted-all-innerneg/checkpoint-${STEP}
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
    --num_train_epochs 1 \
    --load_save_pretrain \
    --pretrain_model_dir ${PRETRAIN_MODEL_PATH}\
    2>&1 | tee ~/v-wanzho/ODQA/code/advance-qa/evidence_chain/pretrain_logs/ft-pretrain-${PREFIX}-${step}.log

    export TEST_DATA_PATH=dev_gttb_candidate_ec_weighted.json
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run_classifier.py \
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
      --test_result_dir ${DATA_PATH}/../scored_chain/pretrained/dev_roberta_base_scored_ec_addtab_weighted_${PREFIX}_pretrain${STEP}.json
    python ~/v-wanzho/ODQA/code/advance-qa/evidence_chain/extraction/evaluate_ranked_evidence_chain.py pretrained/dev_roberta_base_scored_ec_addtab_weighted_${PREFIX}_pretrain${STEP}.json 2>&1 | tee ~/v-wanzho/ODQA/code/advance-qa/evidence_chain/pretrain_logs/evaluate_performance_${PREFIX}_${STEP}.log
    sleep 30
done



