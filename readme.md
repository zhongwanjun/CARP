# CARP
This repository serves primarily as codebase and data, model for training, evaluation and inference of the framework CARP, which is proposed by the paper
[Reasoning over Hybrid Chain for Table-and-Text Open Domain Question Answering](https://arxiv.org/pdf/2201.05880.pdf) is the chain-centric reasoning and pre-training framework for table-and-text open domain QA. 

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

## Data Preprocess

#### Preprocess training data for retrieval

```
python retriever_preprocess.py \
    --split train \
    --nega intable_contra
python retriever_preprocess.py \
    --split dev \
    --nega intable_contra
```

If you want to use the linked passage from BLINK, you can first download the linked passages from [The all_constructed_blink_tables.json](https://github.com/zhongwanjun/CARP/releases/tag/blink-linked-table), then move the json file to `./data_wikitable`. After that, use the following command to preprocess.

```
python retriever_preprocess.py \
    --split train \
    --nega intable_contra \
    --replace_link_passages \
    --aug_blink
python retriever_preprocess.py \
    --split dev \
    --nega intable_contra \
    --replace_link_passages \
    --aug_blink
```

#### Build retrieval corpus

```
python corpus_preprocess.py --split table_corpus_blink
```

This script creates corpus data used for inference.

#### Train retriever

``````
RUN_ID=0
BASIC_PATH=.
DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval
TRAIN_DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval/train_intable_contra_blink_row.pkl
DEV_DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval/dev_intable_contra_blink_row.pkl
MODEL_PATH=${BASIC_PATH}/models/otter
TABLE_CORPUS=table_corpus_blink
mkdir ${MODEL_PATH}

cd retriever/
python train_1hop_tb_retrieval.py \
  --do_train \
  --prefix ${RUN_ID} \
  --predict_batch_size 800 \
  --model_name roberta-base \
  --shared_encoder \
  --train_batch_size 64 \
  --fp16 \
  --max_c_len 512 \
  --max_q_len 70 \
  --metadata \
  --num_train_epochs 20 \
  --accumulate_gradients 1 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.1 \
  --train_file ${TRAIN_DATA_PATH} \
  --predict_file ${DEV_DATA_PATH}  \
  --output_dir ${MODEL_PATH}
``````

#### Inference

##### Step 1: Encode table corpus and dev. questions

```
python encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --metadata \
    --fp16 \
    --max_c_len 512 \
    --predict_file ${BASIC_PATH}/data_ottqa/dev.json \
    --init_checkpoint ${MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${MODEL_PATH}/indexed_embeddings/question_dev
```

Encode table-text block corpus. It takes about 3 hours to encode.

```
python encode_corpus.py \
    --do_predict \
    --encode_table \
    --metadata \
    --predict_batch_size 1600 \
    --model_name roberta-base \
    --fp16 \
    --max_c_len 512 \
    --predict_file ${DATA_PATH}/${TABLE_CORPUS}.pkl \
    --init_checkpoint ${MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}
```

##### Step 4-2: Build index and search with FAISS

The reported results are table recalls.

```
python eval_ottqa_retrieval.py \
    --raw_data_path ${BASIC_PATH}/data_ottqa/dev.json \
    --eval_only_ans \
    --query_embeddings_path ${MODEL_PATH}/indexed_embeddings/question_dev.npy \
    --corpus_embeddings_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
    --id2doc_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
     --output_save_path ${MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
     --beam_size 100
```

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
Eval the model with checkpoint:
cd qa_evidence_chain/
bash test_qa_evidence_chain_retrieved.sh
```
# Data Information
| File Type | File Name | File Location | 
| ---- | ---- | ---- |
| Source Corpus | all_passages.json (and)  all_plain_tables.json | source_corpus/OTT-QA/
| Wikipedia tables and passages | all_tables.json | source_corpus/Wikipedia-table-passages
| Retrieval Results | train/dev/test_output_k100_table_corpus_metagptdoc.json | retrieval_results/
| QA data with extracted evidence chain | train/dev_ranked_evidence_chain_for_qa_weighted.json / test_evidence_chain_weighted_scores.json | qa_with_evidence_chain

[//]: # "| evidence chain pretrain/train/valid/test data | &#40;for-pretraining&#41; bart_output_for_pretraining / &#40;for training&#41; ground-truth-based / &#40;for testing&#41; retrieval_based | evidence_chain_data/ "

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
