# CARP
This repository serves primarily as codebase and data, model for training, evaluation and inference of the framework CARP.
[CARP](https://arxiv.org/pdf/2201.05880.pdf) is the chain-centric reasoning and pre-training framework for table-and-text open domain QA. 
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
export MODEL_PATH=./ODQA/data/retrieval_results
python ../preprocessing/qa_preprocess.py \
  --split dev \
  --reprocess \
  --add_link \
  --topk_tbs ${CONCAT_TBS} \
  --retrieval_results_file ${MODEL_PATH}/dev_output_k100_${TABLE_CORPUS}.json \
  --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
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
export MODEL_PATH=./ODQA/data/retrieval_results
python qa_preprocess.py \
  --split dev \
  --reprocess \
  --add_link \
  --topk_tbs ${CONCAT_TBS} \
  --retrieval_results_file ${MODEL_PATH}/dev_output_k100_${TABLE_CORPUS}.json \
  --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
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
### CARP QA Model Testing
```angular2html
Testing the model with checkpoint:
cd qa_evidence_chain/
bash test_qa_evidence_chain_retrieved.sh
```
# Data Information
| File Type | File Name | File Location | 
| ---- | ---- | ---- |
| Source Corpus | all_passages.json (and)  all_plain_tables.json | source_corpus/OTT-QA/
| Wikipedia tables and passages | all_tables.json | source_corpus/Wikipedia-table-passages
| Retrieval Results | train/dev/test_output_k100_table_corpus_metagptdoc.json | retrieval_results/
| Basic QA data | train/dev/test_preprocessed_table_corpus_metagptdoc_k100cat15.json | basic_qa_data/
| evidence chain pretrain/train/valid/test data | (for-pretraining) bart_output_for_pretraining / (for training) ground-truth-based / (for testing) retrieval_based | evidence_chain_data/ 
| QA data with extracted evidence chain | train/dev_ranked_evidence_chain_for_qa_weighted.json / test_evidence_chain_weighted_scores.json | qa_with_evidence_chain

# Citation
If you find this resource useful, please cite the paper introducing CARP:

```
@article{zhong2022reasoning,
  title={Reasoning over Hybrid Chain for Table-and-Text Open Domain QA},
  author={Zhong, Wanjun and Huang, Junjie and Liu, Qian and Zhou, Ming and Wang, Jiahai and Yin, Jian and Duan, Nan},
  journal={arXiv preprint arXiv:2201.05880},
  year={2022}
}
```
