# Table-ODQA
# Preprocessing

## Requirements

```
nltk
fuzzywuzzy
sklearn
```

## Obtain Data 

Download OTT-QA and wiki tables:

```shell
git clone https://github.com/wenhuchen/OTT-QA.git
cd OTT-QA/data
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ..
cp -r OTT-QA/data data_wikitable
cp -r OTT-QA/released_data data_ottqa
```
# Retrieval

# Evidence Chain
## Data Preprocess
```angular2html
cd preprocess/
export CONCAT_TBS=15
export TABLE_CORPUS=table_corpus_metagptdoc
export MODEL_PATH=/home/t-wzhong/table-odqa-data-model/Data/retrieval_output4qa/shared_roberta_threecat_basic_mean_one_query
python ../preprocessing/qa_preprocess.py \
  --split dev \
  --reprocess \
  --add_link \
  --topk_tbs ${CONCAT_TBS} \
  --retrieval_results_file ${MODEL_PATH}/dev_output_k100_${TABLE_CORPUS}.json \
  --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}_wanjun.json \
  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_dev_k100cat${CONCAT_TBS}.log;
```
## Extraction Model Training
### Extract ground-truth evidence chain
```angular2html
cd evidence_chain/extraction
1. first extract keywords for ground-truth/retrieved table blocks for train/dev set 
    python extract_evidence_chain.py --split train/dev --extract_keywords --kw_extract_type ground-truth/retrieved
2. extract ground-truth evidence chain
    python extract_evidence_chain.py --split train/dev --extract_evidence_chain
   or extract ground-truth evidence chain & generate data for training bart generator
    python extract_evidence_chain.py --split train/dev --extract_evidence_chain --save_bart_training_data
```
### Extract candidate evidence chain
```angular2html
python extract_evidence_chain.py --split train/dev --extract_candidate_evidence_chain
```
### Training Extraction Model
```angular2html
cd evidence_chain/fine-tune
bash run_evidence_train.sh
```
### Evaluate ranked evidence chain by their score
```angular2html
python evaluate_ranked_evidence_chain.py
```
## Extraction Model Pre-training
### Data Preprocess
```angular2html
cd evidence_chain/pretrain_data_process
generate inference data for bart
    python parse_table_psg_link.py bart_inference_data
(generate templated fake pre-train data
    python parse_table_psg_link.py fake_pretrain_data)
```
### BART-based Generator
```angular2html

```
### Pre-training
```angular2html
cd evidence_chain/pretrain
bash run_evidence_pretrain.sh
```
# QA
## Baseline
### Data Preprocess
```angular2html
cd preprocess/
export CONCAT_TBS=15
export TABLE_CORPUS=table_corpus_metagptdoc
export MODEL_PATH=/home/t-wzhong/table-odqa-data-model/Data/retrieval_output4qa/shared_roberta_threecat_basic_mean_one_query
python ../preprocessing/qa_preprocess.py \
  --split dev \
  --reprocess \
  --add_link \
  --topk_tbs ${CONCAT_TBS} \
  --retrieval_results_file ${MODEL_PATH}/dev_output_k100_${TABLE_CORPUS}.json \
  --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}_wanjun.json \
  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_dev_k100cat${CONCAT_TBS}.log;
```
### Baseline QA Model Training
```
cd qa_baseline/
bash train_qa_baseline.sh
```
### CARP QA Model Training
```angular2html
#merge ground-truth and retrieved evidence chains for question answering
cd preprocess
python merge_ec_file.py
cd qa_evidence_chain/
bash train_qa_evidence_chain_retrieved.sh
```
# Data Information
/home/t-wzhong/v-wanzho/ODQA/cleaned_data_code_model
| File Type | File Name | File Location |
| ---- | ---- | ---- |
| Table Corpus | 
