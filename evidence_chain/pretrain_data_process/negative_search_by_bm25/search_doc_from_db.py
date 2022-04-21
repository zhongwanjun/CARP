import os
from tqdm import tqdm
from elastic_search import MyElastic, SearchQuery
# from utils.config import args
# from utils.rule_pattern import RulePattern
# from utils import common
import sys
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import json
sys.path.append('../../')
from qa.utils_qa import read_jsonl
# spacy.download('en_core_web_lg')
ES = MyElastic()

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
def get_result(query, res):
    ranked = res['hits']['hits']
    end = min(len(ranked), 5)
    negs = [item['_source']['text'] for item in ranked[:end] if query != item['_source']['text']]
    return negs

def search_from_db_single(data):
    try:
        res = ES.search_by_chain(data['ec'])
        neg_chain = get_result(data['ec'],res)
        # data['neg_chains'] = neg_chain
        return {'question':data['output'],'chain':data['ec'],'neg_chains':neg_chain,'tb':data['tb']}
    except Exception as e:
        # print(query)
        print(e)
        return None


def search_from_db_pretrain(all_data,outf):
    def get_result(query,res):
        ranked = res['hits']['hits']
        end = min(len(ranked),5)
        negs = [item['_source']['text'] for item in ranked[:end] if query!=item['_source']['text']]
        return negs
    error_cnt,all_cnt=0,0
    for did, data in tqdm(enumerate(all_data)):
        all_cnt += 1
        try:
            res = ES.search_by_chain(data['ec'])
            neg_chain = get_result(data['ec'],res)
            data['neg_chains'] = neg_chain
            if did<=5:
                print(data['ec'])
                print(neg_chain)
                print('---------------------')
            outf.write(json.dumps(data)+'\n')
        except Exception as e:
            # print(query)
            print(e)
            error_cnt += 1
    error_cnt = 0
    all_cnt = 0
    print(error_cnt,all_cnt,error_cnt/all_cnt)
    return tbib2_search_doc

def get_result_finetune(query,res):
    ranked = res['hits']['hits']
    end = min(len(ranked),5)
    negs = [item['_source']['chain'] for item in ranked[:end] if query!=item['_source']['text']]
    return negs
def search_from_db_ft_single(data):
    for tbid, tb in enumerate(data['positive_table_blocks']):
        data['positive_table_blocks'][tbid]['evidence_chain']['es_negative']=[]
        for ecid, ec in enumerate(tb['evidence_chain']['positive']):
            ec_rep = convert_chain_to_string(ec)
            data['positive_table_blocks'][tbid]['evidence_chain']['es_negative'].append([])
            try:
                res = ES.search_by_chain(ec_rep)
                neg_chain = get_result_finetune(ec_rep, res)
                data['positive_table_blocks'][tbid]['evidence_chain']['es_negative'][ecid] = neg_chain
            except Exception as e:
                print(e)
                data['positive_table_blocks'][tbid]['evidence_chain']['es_negative'][ecid] = None
    return data

def search_from_db_finetune(all_data):
    def get_result(query,res):
        ranked = res['hits']['hits']
        end = min(len(ranked),5)
        negs = [item['_source']['chain'] for item in ranked[:end] if query!=item['_source']['text']]
        return negs
    error_cnt,all_cnt=0,0
    for did, data in tqdm(enumerate(all_data)):

        for tbid,tb in enumerate(data['positive_table_blocks']):
            for ecid, ec in enumerate(tb['evidence_chain']['positive']):
                ec_rep = convert_chain_to_string(ec)
                all_cnt += 1
                try:
                    res = ES.search_by_chain(convert_chain_to_string(ec))
                    neg_chain = get_result(ec_rep, res)
                    all_data[did]['positive_table_blocks'][tbid]['evidence_chain']['es_negative'] = neg_chain
                except Exception as e:
                    print(e)
                    error_cnt +=1

    print(error_cnt,all_cnt,error_cnt/all_cnt)
    return tbib2_search_doc
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()
    if args.pretrain:
        basic_dir = './ODQA/data/evidence_chain_data/bart_output_for_pretraining/pre-training/evidence_output_pretrain_shortest.json'
        file_path = os.path.join(basic_dir,'')
        inf = open(basic_dir,'r')
        data = []
        cnt = 0
        for line in inf:
            cnt += 1
            if cnt <= 1500000:
                continue
            data.append(json.loads(line.strip()))

        # data = [json.loads(line.strip()) for line in open(basic_dir,'r').readlines()]
        n_threads = 20
        with open('./ODQA/data/evidence_chain_data/bart_output_for_pretraining/add_negatives/pre-training/evidence_output_pretrain_shortest_esnegs-2.json', 'w') as outf:
            # results = search_from_db_pretrain(data,outf)
            running_function = search_from_db_single
            with Pool(n_threads) as p:
                func_ = partial(running_function)
                all_results = list(tqdm(p.imap(func_, data, chunksize=16), total=len(data),
                                        desc="find negatives", ))
            for result in all_results:

                if result:
                    outf.write(json.dumps(result) + '\n')
    if args.finetune:
        basic_dir = './ODQA/data/evidence_chain_data/ground-truth-based/ground-truth-evidence-chain/'
        file_paths = [os.path.join(basic_dir, file) for file in
                      ['train_gt-ec-weighted.json', 'dev_gt-ec-weighted.json']]
        n_threads = 20
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf8') as inf:
                data=json.load(inf)
                # data = search_from_db_finetune(data)
                # outf = open(file_path.replace('.json','-esneg.json'),'w',encoding='utf8')
                # print('Saving output to {}'.format(file_path.replace('.json','esneg.json')))
                # del data
                running_function = search_from_db_ft_single
                with Pool(n_threads) as p:
                    func_ = partial(running_function)
                    all_results = list(tqdm(p.imap(func_, data, chunksize=16), total=len(data),
                                            desc="find negatives", ))
                print([item['positive_table_blocks'] for item in all_results[:2]])
                outf = open(file_path.replace('.json', '-esneg.json'), 'w', encoding='utf8')
                print('Saving output to {}'.format(file_path.replace('.json','-esneg.json')))
                json.dump(all_results,outf)
                # for result in all_results:
                #     if result:
                #         outf.write(json.dumps(result) + '\n')

