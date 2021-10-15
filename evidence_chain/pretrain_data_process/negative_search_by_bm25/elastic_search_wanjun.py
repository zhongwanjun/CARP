import json
import os
import re
import unicodedata

import inflect
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from tqdm import tqdm
import sys
sys.path.append('../')
# from utils_preprocess import args
# from utils.rule_pattern import RulePattern

inflect = inflect.engine()
# RP = RulePattern()

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
def check_contain_upper(self, password):
    pattern = re.compile('[A-Z]+')
    match = pattern.findall(password)
    if match:
        return True
    else:
        return False


class SearchQuery():
    @classmethod
    def claim2text(cls, claim, type='title_text'):
        search_body = {
            "query": {
                "match": {
                    type: claim
                }
            }
        }
        return search_body

    @classmethod
    def claim2text_title(cls, claim):
        # score in both text and title
        search_body = {
            "query": {
                "multi_match": {
                    "query": claim,
                    "fields": ['text', 'title'],
                    "fuzziness": "AUTO"
                }
            }
        }
        return search_body

    @classmethod
    def kws2title(cls, multi_claim):
        search_body = {
            "query": {
                "bool": {
                    "should": [

                    ]
                }
            }}
        for claim in multi_claim:
            tiny_body = {
                "match_phrase": {
                    "title": {
                        'query': claim,
                        "slop": 2
                    }

                    # "slop": 5
                }
            }
            search_body['query']['bool']['should'].append(tiny_body)
        return search_body


class MyElastic():
    def __init__(self, index_name='evidence_chain'):
        self.es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])
        self.index_name = index_name
        body = {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer":"analyzed"
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name,request_timeout=60):
            self.es.indices.create(self.index_name,request_timeout=60)
            self.es.indices.put_mapping(index=self.index_name, doc_type='evidence_chain',
                                        body=body, include_type_name=True)
            # self.es.indices.put_mapping(index=self.index_name, doc_type='wiki_sentence',
            #                             body=body, include_type_name=True)

    def search(self, search_body):
        ret = self.es.search(index=self.index_name, body=search_body, size=10)
        return ret

    def bulk_insert_all_chains_finetune(self, file_paths):
        data = []
        for file_path in file_paths:
            with open(file_path,'r',encoding='utf8') as inf:
                data.extend(json.load(inf))
        all_chains_strs = []
        all_chains = []
        for item in data:
            for table_block in item['positive_table_blocks']:
                ecs = table_block['evidence_chain']['positive']
                ec_strs = [convert_chain_to_string(chain) for chain in ecs]
                all_chains_strs.extend(ec_strs)
                all_chains.extend(ecs)
        cnt = 0
        actions = []
        for id,chain in tqdm(enumerate(all_chains_strs)):
            input_body = {
                "_index": self.index_name,
                "_type":"evidence_chain",
                "_id":id,
                "_source":{
                    'id':id,
                    'text':chain,
                    'chain':all_chains[id]
                }
            }
            cnt+=1
            actions.append(input_body)
        if len(actions) != 0:
            print(helpers.bulk(self.es, actions,request_timeout=60))

    def bulk_insert_all_chains_pretrain(self, file_path):
        with open(file_path,'r',encoding='utf8') as inf:
            all_chains = [json.loads(line.strip())['ec'] for line in inf.readlines()]
        cnt = 0
        actions = []
        for id,chain in tqdm(enumerate(all_chains)):
            input_body = {
                "_index": self.index_name,
                "_type":"evidence_chain",
                "_id":id,
                "_source":{
                    'id':id,
                    'text':chain,
                }
            }
            cnt+=1
            actions.append(input_body)
        if len(actions) != 0:
            print(helpers.bulk(self.es, actions,request_timeout=60))




    def delete_one(self):
        res = self.es.indices.delete(index='evidence_chain',request_timeout=60)
        print(res)

    def create(self):

        body = {
            "properties": {
                "text": {
                    "type": "text",
                    # "analyzer": "analyzed"
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(self.index_name)
            self.es.indices.put_mapping(index=self.index_name, doc_type='evidence_chain',
                                        body=body, include_type_name=True)

    def clear_cache(self, index_name='wiki_search'):
        res = self.es.indices.clear_cache(index=index_name)
        print(res)

    def delete_index(self, index_name="wiki_search"):
        query = {'query': {"match_all": {}}}
        res = self.es.delete_by_query(index=index_name, body=query)
        print(res)

    def search_by_chain(self,query):
        search_body = SearchQuery.claim2text(query,'text')
        ret = self.search(search_body)
        return ret



if __name__ == "__main__":
    ES = MyElastic()
    # ES.delete_one()
    res = ES.search_by_id('Soul_Food_-LRB-film-RRB-0')
    print(res)
    # ES.delete_index()
    # fp = '/mnt/wanjun/FEVER/data/wiki-pages/wiki-001.jsonl'
    # ES.bulk_insert_one(fp)
    # claim = 'football'
    # ES.search(claim)
