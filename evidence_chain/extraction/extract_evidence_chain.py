import sys

sys.path.append('../../')
sys.path.append('../')
import argparse
import logging
import json
import pickle
import pandas as pd
import os
import random
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from extraction.evidence_chain_addpathscore import *#, extract_evidence_chain, extract_key_words
from utils.common import convert_tb_to_string_metadata
import sys

import argparse
import logging
import json
import pickle
import os
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

basic_dir = '/home/t-wzhong/v-wanzho/ODQA/data'
# basic_dir = '..'
resource_path = f'{basic_dir}/data_wikitable/'

def convert_chain_to_string(chain):
    chain_rep = []
    for node in chain[1:]:
        # print(node)
        if node['origin']['where'] == 'table':
            prefix = '[TAB] '
        elif node['origin']['where'] == 'passage':
            prefix = '[PASSAGE] {} : '.format(node['origin']['index'].replace('/wiki/', '').replace('_', ' ').split('/')[0])
        else:
            prefix = '[QUESTION] '
        chain_rep.append(prefix+node['content'])
    # print(' [TO] '.join(chain_rep))
    # input()
    return ' ; '.join(chain_rep)
def merge_gt_linked_passages(gt_passages, linked_passages):
    if len(linked_passages) <= len(gt_passages):
        return gt_passages
    elif not gt_passages:
        return linked_passages
    else:
        gt_ids = [gt['passage_id'] for gt in gt_passages]
        neg_passages = [item for item in linked_passages if item['passage_id'] not in gt_ids]
        output = gt_passages + random.choices(neg_passages,
                                              k=min(len(neg_passages), len(linked_passages) - len(gt_passages)))
        return output


def extract_keywords_api(data,RP):

    qkws = extract_question_kws(data['question'],RP)
    data['question_keywords'] = qkws
    # data['positive_table_blocks'] = extract_key_words_multiple(data['positive_table_blocks'], RP)
    for tb_id,table_block in enumerate(data['positive_table_blocks']):
        # evidence_chain = extract_evidence_chain(data['question'],data['answer-text'],table_block['table_segment'],table_block['gt_passages'],data['where'])
        passages_kws, table_v_kws = extract_key_words(table_block['table_segment'],table_block['gt_passages'],RP)
        data['positive_table_blocks'][tb_id]['gt_passages_keywords'] = passages_kws
        data['positive_table_blocks'][tb_id]['table_value_keywords'] = table_v_kws
    del data['positive_passages']
    del data['retrieved_tbs']
    return data

def extract_keywords_retrieved_api(data,RP):
    qkws = extract_question_kws(data['question'], RP)
    data['question_keywords'] = qkws
    for tb_id, table_block in enumerate(data['retrieved_tbs'][:15]):
        # evidence_chain = extract_evidence_chain(data['question'],data['answer-text'],table_block['table_segment'],table_block['gt_passages'],data['where'])
        passages_kws, table_v_kws = extract_key_words_retrived_tb(table_block['table'], table_block['passages'], RP)
        data['retrieved_tbs'][tb_id]['passage_keywords'] = passages_kws
        data['retrieved_tbs'][tb_id]['table_value_keywords'] = table_v_kws

    return data



def extract_evidence_chain_api(zip_data):
    orig_data, data = zip_data
    # data = zip_data
    error_count = 0
    chain_length = 0
    for tb_id, table_block in enumerate(data['positive_table_blocks']):
    # return processed_data, qa_data
        pid2cellid = dict([(item['passage_id'],item['link_table']) for item in data['positive_table_blocks'][tb_id]['gt_passages']])
        for idx in range(len(table_block['gt_passages'])):
            table_block['gt_passages'][idx]['link_table'] = pid2cellid[table_block['gt_passages'][idx]['passage_id']]
        evidence_chains, negative_evidence_chain, all_ec = extract_evidence_chain(data,table_block)

        length = sum([len(chain) for chain in evidence_chains])/len(evidence_chains) if evidence_chains else 0
        chain_length += length
        if not evidence_chains:
            error_count+=1
        data['positive_table_blocks'][tb_id]['context'] = orig_data['positive_table_blocks'][tb_id]['context']
        data['positive_table_blocks'][tb_id]['evidence_chain'] = {'positive':evidence_chains, 'negative': negative_evidence_chain,'all':all_ec}
    # del data['retrieved_tbs']
    # del data['table']
    # del data['positive_passages']
    return data, error_count/len(data['positive_table_blocks']),chain_length/len(data['positive_table_blocks'])

def extract_evidence_chain_eval_api(zip_data):
    orig_data, data = zip_data
    error_count = 0
    for tb_id, table_block in enumerate(data['positive_table_blocks']):
    # return processed_data, qa_data
        pid2cellid = dict([(item['passage_id'],item['link_table']) for item in orig_data['positive_table_blocks'][tb_id]['gt_passages']])
        for idx in range(len(table_block['gt_passages'])):
            table_block['gt_passages'][idx]['link_table'] = pid2cellid[table_block['gt_passages'][idx]['passage_id']]
        candidate_evidence_chains = extract_evidence_chains_for_ranking(data,table_block)
        if not candidate_evidence_chains:
            error_count+=1
        data['positive_table_blocks'][tb_id]['candidate_evidence_chains'] = candidate_evidence_chains#{'positive':evidence_chains, 'negative': negative_evidence_chain}
    # del data['retrieved_tbs']
    # del data['table']
    # del data['positive_passages']
    return data
def extract_evidence_chain_eval_retrieved_api(zip_data):
    data = zip_data
    error_count = 0
    for tb_id, table_block in enumerate(data['retrieved_tbs']):
    # return processed_data, qa_data
    #     pid2cellid = dict([(item['passage_id'],item['link_table']) for item in orig_data['positive_table_blocks'][tb_id]['gt_passages']])
    #     for idx in range(len(table_block['chain']['context'])):
    #         table_block['passages'][idx]['link_table'] = pid2cellid[table_block['gt_passages'][idx]['passage_id']]
        candidate_evidence_chains = extract_evidence_chains_for_ranking_retrived(data,table_block)
        if not candidate_evidence_chains:
            error_count+=1
        data['retrieved_tbs'][tb_id]['candidate_evidence_chains'] = candidate_evidence_chains#{'positive':evidence_chains, 'negative': negative_evidence_chain}
    return data
def save_pretrain_data(all_results,evi_chain_save_path):
    shortest_outputs = []
    all_path_output = []
    for idx, data in enumerate(all_results):
        # print(data)
        question = data['question']
        for tb_id, table_block in enumerate(data['positive_table_blocks']):
            shorest_chains = table_block['evidence_chain']['positive']
            for chain in shorest_chains:
                positive_chain = convert_chain_to_string(chain)
                table_context = table_block['context']
                shortest_outputs.append({'input':f'{table_context} [EC] {positive_chain}','output':question})
            positive_chains = table_block['evidence_chain']['all']
            for chain in positive_chains:
                positive_chain = convert_chain_to_string(chain)
                table_context = table_block['context']
                all_path_output.append({'input':f'{table_context} [EC] {positive_chain}','output':question})
    print(f'Saving to {evi_chain_save_path}')
    with open(evi_chain_save_path.replace('bart','bart_shortest'),'w',encoding='utf8') as outf:
        for line in shortest_outputs:
            outf.write(json.dumps(line)+'\n')
    with open(evi_chain_save_path.replace('bart','bart_all'),'w',encoding='utf8') as outf:
        for line in all_path_output:
            outf.write(json.dumps(line)+'\n')
if __name__ == '__main__':
    """
        export CONCAT_TBS=15
        python qa_preprocess_gptlink.py \
                --split dev \
                --topk_tbs ${CONCAT_TBS} \
                --retrieval_results_file ${MODEL_PATH}/indexed_embeddings/dev_output_k100.json \
                --qa_save_path ${MODEL_PATH}/dev_preprocessed_gtmodify_k100cat${CONCAT_TBS}.json \
                2>&1 |tee ${MODEL_PATH}/run_logs/preprocess_qa_dev_gtmodify_k100cat${CONCAT_TBS}.log

        python qa_preprocess_gptlink.py \
            --split dev \
            --retrieval_results_file ../preprocessed_data/qa_evidence_chain/data4qa_v2/dev_output_k100_tfidf20.json

        python qa_preprocess_gptlink.py \
            --split train \
	        --retrieval_results_file ../preprocessed_data/qa_evidence_chain/data4qa_v2/train_output_k100.json
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--split', default='train', type=str, choices=['train', 'dev','test'])
    parser.add_argument('--topk_tbs', default=15, type=int)
    # parser.add_argument('--use_gpt_link_passages',action='store_true')
    parser.add_argument('--kw_extract_type', default='ground-truth', type=str, choices=['ground-truth','retrieved'])
    parser.add_argument('--reprocess', action='store_true')
    parser.add_argument('--extract_keywords', action='store_true')
    parser.add_argument('--save_bart_training_data', action='store_true')
    parser.add_argument('--extract_evidence_chain', action='store_true')
    parser.add_argument('--extract_candidate_evidence_chain', action='store_true')
    parser.add_argument("--qa_save_path", type=str,
                        default=f'{basic_dir}/preprocessed_data/qa_evidence_chain/train_preprocessed_gtmodify_normalized_newretr.json', )
    parser.add_argument('--part', default=0, type=int, choices=[0,1,2,3])
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    n_threads = 20
    if args.split in ['train', 'dev','test']:
        print("using {}".format(args.split))
        # table_path = 'traindev_tables_tok'
        # request_path = 'traindev_request_tok'
        # qa_save_path = f'{basic_dir}/preprocessed_data/qa_evidence_chain/{args.split}_preprocessed_normalized_gtmodify_top100.json'
        qa_save_path = f'/home/t-wzhong/table-odqa/Data/retrieval_output4qa/shared_roberta_threecat_basic_mean_one_query/{args.split}_preprocessed_table_corpus_metagptdoc_k100cat15_wanjun.json'
        keywords_save_path = f'/home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/evidence_chain/retrieval-based/keywords/new_chain_blink_{args.split}_keywords.json'#_{args.part}'
        # keywords_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/ground-truth-based/keywords/{args.split}_gt-tb-keywords.json'
        # evi_chain_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/{args.split}_preprocessed_normalized_gtmodify_candidate_evichain_addnoun.json'
        # evi_chain_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/retrieval-based/candidate_chain/{args.split}_evidence_chain_weighted_ranking_{args.part}.json'
        # pretrain_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/pre-training/{args.split}_ec_path_for_bart_weighted.jsonl'
        evi_chain_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/ground-truth-based/ground-truth-evidence-chain/{args.split}_gt-ec-simple_multineg.json'
        # cand_evi_chain_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/ground-truth-based/candidate_chain/{args.split}_gttb_all_candidate_ec_simple.json'
        cand_evi_chain_save_path = f'{basic_dir}/preprocessed_data/evidence_chain/retrieval-based/candidate_chain/{args.split}_evidence_chain_simple_ranking.json'#_{args.part}.json'
        if args.extract_keywords:
            RP = RulePattern(gpu=0)
            with open(qa_save_path, 'r') as f:
                data = json.load(f)
            all_results = []
            for item in tqdm(data,desc="extract keywords"):
                if args.kw_extract_type=='ground-truth':
                    all_results.append(extract_keywords_api(item, RP))
                elif args.kw_extract_type=='retrieved':
                    all_results.append(extract_keywords_retrieved_api(item,RP))
            # running_function = extract_evidence_chain_api
            # with Pool(n_threads) as p:
            #     func_ = partial(running_function)
            #     all_results = list(tqdm(p.imap(func_, data, chunksize=16), total=len(data),
            #                             desc="extract evidence chain", ))
                # qa_results = [res['qa_data'] for res in all_results]
                # hits = [res['hit'] for res in all_results]
                # logger.info("{}, {};".format(len(all_results), len(qa_results)))


            with open(keywords_save_path, 'w') as f:
                json.dump(all_results, f, indent=4)
                print('Saving the output to {}'.format(keywords_save_path))
        # print(json.dumps(all_results[0]['positive_table_blocks'], indent=4))
        if args.extract_evidence_chain:
            with open(qa_save_path, 'r') as f:
                orig_data = json.load(f)
            with open(keywords_save_path, 'r') as f:
                data = json.load(f)

            # all_results = []
            ziped_data = list(zip(orig_data, data))#[:100]
            # for item in tqdm(ziped_data, desc="extract evidence chain"):
            #     all_results.append(extract_evidence_chain_api(item))
            # ziped_data = data#zip(data)
            # running_function = extract_evidence_chain_eval_api
            running_function = extract_evidence_chain_api
            with Pool(n_threads) as p:
                func_ = partial(running_function)
                all_results = list(tqdm(p.imap(func_, ziped_data, chunksize=16), total=len(data),
                                        desc="extract evidence chain", ))
            chain_length = [res[2] for res in all_results]
            error_counts = [res[1] for res in all_results]
            all_results = [res[0] for res in all_results]
            # chain_length = [res[2] for res in all_results]
            print('{} of the table blocks that can not find paths'.format(sum(error_counts)/len(data)))
            print('average path length {}'.format(sum(chain_length) / len(data)))
            if args.save_bart_training_data:
                save_pretrain_data(all_results,pretrain_save_path)
            with open(evi_chain_save_path, 'w') as f:
                json.dump(all_results, f, indent=4)
                print('Saving the output to {}'.format(evi_chain_save_path))

        if args.extract_candidate_evidence_chain:
            # with open(qa_save_path, 'r') as f:
            #     orig_data = json.load(f)
            with open(keywords_save_path, 'r') as f:
                data = json.load(f)

            # ziped_data = list(zip(orig_data, data))#[:100]
            ziped_data = data
            running_function = extract_evidence_chain_eval_retrieved_api
            with Pool(n_threads) as p:
                func_ = partial(running_function)
                all_results = list(tqdm(p.imap(func_, ziped_data, chunksize=16), total=len(data),
                                        desc="extract evidence chain", ))

            with open(cand_evi_chain_save_path, 'w') as f:
                json.dump(all_results, f)
                print('Saving the output to {}'.format(cand_evi_chain_save_path))


    else:
        raise NotImplementedError


