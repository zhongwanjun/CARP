export CONCAT_TBS=15
export TABLE_CORPUS=table_corpus_metagptdoc
export MODEL_PATH=./ODQA/retrieval_results/shared_roberta_threecat_basic_mean_one_query
#python ../preprocessing/qa_preprocess.py \
#  --split dev \
#  --reprocess \
#  --add_link \
#  --topk_tbs ${CONCAT_TBS} \
#  --retrieval_results_file ${MODEL_PATH}/dev_output_k100_${TABLE_CORPUS}.json \
#  --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
#  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_dev_k100cat${CONCAT_TBS}.log;
python ../preprocessing/qa_preprocess.py \
  --split train \
  --reprocess \
  --add_link \
  --topk_tbs ${CONCAT_TBS} \
  --retrieval_results_file ${MODEL_PATH}/train_output_k100_${TABLE_CORPUS}.json \
  --qa_save_path ${MODEL_PATH}/train_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_train_k100cat${CONCAT_TBS}.log;