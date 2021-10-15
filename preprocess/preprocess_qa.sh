#python qa_preprocess.py --split dev --reprocess --retrieval_results_file /home/t-wzhong/table-odqa-data-model/Data/qa_evidence_chain/roberta_meta_training/dev_output_gptdoc_k100_tfidf20.json
#python qa_preprocess.py --split train --reprocess --retrieval_results_file /home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/qa_evidence_chain/data4qa_v2/train_output_k100.json
#python qa_preprocess_blink.py --split dev --reprocess --retrieval_results_file /home/t-wzhong/table-odqa-data-model/Data/retrieval_output4qa/shared_roberta_metagptdoc_woextraposi_normtable_momentum960/dev_output_k100_table_corpus_metagptdoc.json
export CONCAT_TBS=15
export TABLE_CORPUS=table_corpus_metagptdoc
export MODEL_PATH=/home/t-wzhong/table-odqa-data-model/Data/retrieval_output4qa/shared_roberta_threecat_basic_mean_one_query
#python ../preprocessing/qa_preprocess.py \
#  --split dev \
#  --reprocess \
#  --add_link \
#  --topk_tbs ${CONCAT_TBS} \
#  --retrieval_results_file ${MODEL_PATH}/dev_output_k100_${TABLE_CORPUS}.json \
#  --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}_wanjun.json \
#  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_dev_k100cat${CONCAT_TBS}.log;
python ../preprocessing/qa_preprocess.py \
  --split train \
  --reprocess \
  --add_link \
  --topk_tbs ${CONCAT_TBS} \
  --retrieval_results_file ${MODEL_PATH}/train_output_k100_${TABLE_CORPUS}.json \
  --qa_save_path ${MODEL_PATH}/train_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}_wanjun.json \
  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_train_k100cat${CONCAT_TBS}.log;