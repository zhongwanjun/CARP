# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import collections
import json
import random
import pandas as pd
import numpy as np
from transformers.data.processors.utils import DataProcessor
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from torch.utils.data import TensorDataset
import sys

sys.path.append('../')
from preprocessing.utils_preprocess import rank_doc_tfidf
from utils.common import convert_tb_to_string_metadata, convert_tb_to_string

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import os, pickle
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import re
import logging
logger = logging.getLogger(__name__)

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


class OTTQAExample(object):
    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers='',
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]



class OTTQAProcessor(DataProcessor):
    def __init__(self, args):
        super(OTTQAProcessor, self).__init__()
        self.args = args

    def get_qa_data(self, input_data, filename,split='dev'):
        qa_data_path = os.path.join(self.args.data_dir, filename + '_qa')
        if (not os.path.exists(qa_data_path)) or self.args.repreprocess:
            print('using original function')
            qa_data = prepare_qa_data(input_data)
            #with open(qa_data_path, 'w', encoding='utf8') as outf:
             #   json.dump(qa_data, outf)
            #print("dumping qa_evidence_chain data to {}".format(qa_data_path))
        else:
            qa_data = json.load(open(qa_data_path, 'r', encoding='utf8'))
            print("loading qa_evidence_chain data from {}".format(qa_data_path))
        return qa_data

    def get_train_examples(self, filename=None):
        print("loading input data from {}".format(os.path.join(self.args.data_dir, filename)))
        # input_data = readGZip(filename)
        with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        qa_data = self.get_qa_data(input_data, filename,'train')
        print("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
        return self._create_examples(qa_data, "train")

    def get_dev_examples(self, filename=None):
        # input_data = readGZip(filename)
        with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        qa_data = self.get_qa_data(input_data, filename,'dev')
        print("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
        return self._create_examples(qa_data, "dev")

    def _create_examples(self, qa_data, set_type):

        is_training = set_type == "train"
        examples = []
        for entry in tqdm(qa_data, desc='   Creating examples: '):

            # import pdb
            # pdb.set_trace()
            title = entry["title"]
            context_text = entry["context"]
            # for qa_evidence_chain in paragraph["qas"]:
            qas_id = entry["question_id"]
            question_text = entry["question"]
            start_position_character = None
            answer_text = None
            answers = []

            if "is_impossible" in entry:
                is_impossible = entry["is_impossible"]
            else:
                is_impossible = False

            if not is_impossible:
                if is_training:
                    answer = entry["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]
                else:
                    answers = entry["answers"]

            example = OTTQAExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
            )

            examples.append(example)

        return examples

class OTTQAProcessorConcat(OTTQAProcessor):
    def __init__(self, args):
        super(OTTQAProcessorConcat, self).__init__(args)
    def get_qa_data(self, input_data, filename,split='dev'):
        qa_data_path = os.path.join(self.args.data_dir, filename + '_qa_shuffle_{}'.format(self.args.topk_tbs))
        print('using new function')
        # if (not os.path.exists(qa_data_path)) or self.args.repreprocess:
        # qa_data = prepare_qa_data_evidence_chain_ranking(input_data,self.args.topk_tbs)
        qa_data = prepare_qa_data_evidence_chain_ranking_retrieved(input_data,self.args.topk_tbs)
            # with open(qa_data_path, 'w', encoding='utf8') as outf:
            #     json.dump(qa_data, outf)
            # logger.info("dumping qa_evidence_chain data to {}".format(qa_data_path))
        # else:
            # qa_data = json.load(open(qa_data_path, 'r', encoding='utf8'))
            # logger.info("loading qa_evidence_chain data from {}".format(qa_data_path))
        return qa_data

    def get_qa_data_eval(self, input_data,split='dev'):
        qa_data_path = os.path.join(self.args.data_dir, split + '_qa_shuffle_{}'.format(self.args.topk_tbs))
        print('using evaluation function')
        qa_data = prepare_qa_data_evidence_chain_ranking_retrieved_eval(input_data, self.args.topk_tbs)
        return qa_data
    def get_qa_data_test(self, input_data, split='test'):
        qa_data,key2idx = prepare_qa_data_evidence_chain_ranking_retrieved_test(input_data,self.args.topk_tbs)
        return qa_data,key2idx
    def get_train_examples(self, filename=None):
        print("loading input data from {}".format(os.path.join(self.args.data_dir, filename)))
        # input_data = readGZip(filename)
        with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)#[:100]
        qa_data = self.get_qa_data(input_data, filename,'train')
        print("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
        return self._create_examples(qa_data, "train")
    def get_dev_examples(self, filename=None):
        # input_data = readGZip(filename)
        with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)#[:100]
        print('using eval data')
        qa_data = self.get_qa_data_eval(input_data)
        print("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
        return self._create_examples(qa_data, "dev")

    def get_test_examples(self, input_data=None):
        if not input_data:
            with open(os.path.join(self.args.data_dir, self.args.predict_file), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)

        qa_data,key2idx = self.get_qa_data_test(input_data)#prepare_qa_data_tbconcat_test(input_data, self.args.topk_tbs)

        logger.info("input data:length {}/{}".format(len(input_data), len(qa_data)))
        return self._create_examples(qa_data, "test"),key2idx


def convert_chain_to_string(chain):
    chain_rep = []
    for node in chain:
        # print(node)
        if node['origin']['where'] == 'question':
            continue
        if node['origin']['where'] == 'table':
            prefix = '[TAB] '
        elif node['origin']['where'] == 'passage':
            prefix = '[PASSAGE] {} : '.format(node['origin']['index'].replace('/wiki/', '').replace('_', ' ').split('/')[0])

        chain_rep.append(prefix + node['content'])
    # print(' [TO] '.join(chain_rep))
    # input()
    return ' [TO] '.join(chain_rep)
class OTTQAFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignment methods.
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            start_position,
            end_position,
            is_impossible,
            ec_mask,
            qas_id: str = None,
            encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
        self.ec_mask = ec_mask
        self.encoding = encoding


def get_table_block(node, documents):
    table_block = documents[node[2]]
    table_block += 'Title : {} . '.format(node[0]) + table_block
    return table_block


def get_test_data(args):
    with open(args.request_path, 'r') as f:
        requests = json.load(f)

    # evaluate(args, model, tokenizer, prefix=global_step)
    with open(args.predict_file, 'r') as f:
        data = json.load(f)

    full_split = []
    key2idx = {}
    for step, d in enumerate(data):
        if isinstance(d['pred'], str):
            continue
        table_id = d['table_id']
        node = d['pred']
        context = 'Title : {} . {}'.format(node[0], requests[node[1]])
        context = get_table_block(node[0], requests[node[1]])
        full_split.append({'context': context, 'title': table_id,
                           'question': d['question'], 'question_id': d['question_id'],
                           'answers': [{'answer_start': None, 'text': None}]})
        key2idx[d['question_id']] = step

    processor = OTTQAProcessor(args)
    examples = processor._create_examples(full_split, 'dev')

    # logger.info("Preprocessing {} examples".format(len(examples)))
    features, dataset = ottqa_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
        threads=args.threads,
    )
    return dataset, examples, features, key2idx


from fuzzywuzzy import fuzz
def prepare_qa_data_evidence_chain(data, topk_tbs = 15):
    split = []
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        positive_table_block = d['positive_table_blocks']
        # positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        ground_truth_context = []
        gt_tb_ids = {}

        for block in positive_table_block:
            best_score = 0
            selected_chain = ''
            # print( block['evidence_chain']['positive'])
            for chain in block['evidence_chain']['positive']:
                positive_chain = convert_chain_to_string(chain)
                score = fuzz.partial_ratio(d['question']+' '+orig_answer,positive_chain)
                if score>best_score:
                    best_score = score
                    selected_chain = positive_chain
            context = '{} [EVIDENCE_CHAIN] {} [END_EVIDENCE_CHAIN]'.format(block['context'],selected_chain)
            # print(context)
            # input()
            gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])] = context
            ground_truth_context.append(context)

        all_context = [{'passage':c,'type':'ground-truth'} for c in ground_truth_context]# + [{'passage':c,'type':'searched'} for c in searched_context]

        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        input_context_str = '[TB] '+' [TB] '.join(ranked_context)

#         all_start_pos = [orig_answer.start() for orig_answer in re.finditer(orig_answer.lower(), context_str.lower())]
#         context_str = ' [TB] '.join()
        start = input_context_str.lower().find(orig_answer.lower())
        #randomly select a answer if there exists multiple
#         start = random.choice(all_start_pos)
        if start == -1:
            import pdb
            pdb.set_trace()
            while input_context_str[start].lower() != orig_answer[0].lower():
                start -= 1
        answer = input_context_str[start:start + len(orig_answer)]

        # answer = orig_answer
        split.append({'context': input_context_str, 'title': d['table_id'],
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': start, 'text': answer}]})

        if idx < 2:
            logger.info(split[-1])
    # logger.info('{}/{} examples are hitted'.format(len(hit), len(data)))
    return split
from scipy.special import softmax
def find_union_chain(ranked_ec,topt=2):
    nodes = set()
    all_nodes = []
    for ec in ranked_ec[:topt]:
        # print(ec['path'])
        add = [node for node in ec['path'][1:] if node['content'] not in nodes]
        all_nodes.extend(add)
        nodes.update(set([node['content'] for node in ec['path'][1:]]))
    return all_nodes
def prepare_qa_data_evidence_chain_ranking_retrieved(data, topk_tbs = 15):
    split = []
    error_cnt = 0
    all_cnt = 0
    hit = set()
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        positive_table_block = d['positive_table_blocks']
        # positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        ground_truth_context = []
        gt_tb_ids = {}

        for block in positive_table_block:
            all_cnt+=1
            if 'candidate_evidence_chains' in block.keys():
                ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: softmax(k['score'])[1], reverse=True)
            else:
                ranked_ec = []

            # for chain in block['candidate_evidence_chains']:
            #     positive_chain = convert_chain_to_string(chain['path'])
            #     score = softmax(chain['score'])#fuzz.partial_ratio(d['question']+' '+orig_answer,positive_chain)
            #     if score[1]>best_score:
            #         best_score = score[1]
            #         selected_chain = positive_chain
            # print(d['question'],orig_answer)
            # print(selected_chain,best_score)
            # input()
            # if ranked_ec:
            #     for i in range(min(len(ranked_ec),1)):
            #         selected_chain.append(convert_chain_to_string(ranked_ec[i]['path']))
            if ranked_ec:
                if len(ranked_ec) >= 2:
                    union_ec = find_union_chain(ranked_ec, topt=2)
                    selected_chain = convert_chain_to_string(union_ec)
                else:
                    selected_chain = convert_chain_to_string(ranked_ec[0]['path'])
                # selected_chain = convert_chain_to_string(ranked_ec[0]['path'])
            else:
                selected_chain = 'none'
                error_cnt += 1
            context = '{} [EVIDENCE_CHAIN] {} [END_EVIDENCE_CHAIN]'.format(block['context'],'{}'.format(selected_chain))
            # context = block['context']
            gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])] = context
            ground_truth_context.append(context)
        searched_context = []
        for block in d['retrieved_tbs']:
            all_cnt+=1
            if '{}-{}'.format(block['table_id'], block['row_id']) in gt_tb_ids.keys():
                hit.add(d['question_id'])
                continue
            if block['candidate_evidence_chains']:
                ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: k['score'][1], reverse=True)
                if False:#len(ranked_ec)>=2:
                    union_ec = find_union_chain(ranked_ec, topt=2)
                    selected_chain = convert_chain_to_string(union_ec)
                else:
                    selected_chain = convert_chain_to_string(ranked_ec[0]['path'])
            else:
                selected_chain = 'none'
                error_cnt += 1
            context = '{} [EVIDENCE_CHAIN] {} [END_EVIDENCE_CHAIN]'.format(block['context'],selected_chain)
            # context = block['context']
            searched_context.append(context)
        all_context = [{'passage':c,'type':'ground-truth'} for c in ground_truth_context] + [{'passage':c,'type':'searched'} for c in searched_context]
        all_context = all_context[:topk_tbs]
        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        input_context_str = '[TB] '+' [TB] '.join(ranked_context)

#         all_start_pos = [orig_answer.start() for orig_answer in re.finditer(orig_answer.lower(), context_str.lower())]
#         context_str = ' [TB] '.join()
        start = input_context_str.lower().find(orig_answer.lower())
        #randomly select a answer if there exists multiple
#         start = random.choice(all_start_pos)
        if start == -1:
            import pdb
            pdb.set_trace()
            while input_context_str[start].lower() != orig_answer[0].lower():
                start -= 1
        answer = input_context_str[start:start + len(orig_answer)]

        # answer = orig_answer
        split.append({'context': input_context_str, 'title': d['table_id'],
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': start, 'text': answer}]})

        if idx < 2:
            logger.info(split[-1])
    logger.info('{}/{} examples are hitted'.format(len(hit), len(data)))
    logger.info('no candidate rate: {}'.format(error_cnt/all_cnt))
    return split

def prepare_qa_data_evidence_chain_ranking_retrieved_eval(data, topk_tbs = 15):
    split = []
    hit = set()
    error_cnt,all_cnt = 0,0
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        positive_table_block = d['positive_table_blocks']
        # positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        # ground_truth_context = []
        gt_tb_ids = []

        for block in positive_table_block:
            gt_tb_ids.append('{}-{}'.format(block['table_id'],block['row_id']))
        searched_context = []
        for block in d['retrieved_tbs'][:topk_tbs]:
            all_cnt+=1
            if '{}-{}'.format(block['table_id'], block['row_id']) in gt_tb_ids:
                hit.add(d['question_id'])

            if block['candidate_evidence_chains']:
                ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: softmax(k['score'])[1], reverse=True)
                if False:#len(ranked_ec)>=2:
                    union_ec = find_union_chain(ranked_ec, topt=2)
                    selected_chain = convert_chain_to_string(union_ec)
                else:
                    selected_chain = convert_chain_to_string(ranked_ec[0]['path'])
                # selected_chain = convert_chain_to_string(ranked_ec[0]['path'])#ranked_ec[0]['path']
            else:
                selected_chain = 'none'
                error_cnt+=1
            context = '{} [EVIDENCE_CHAIN] {} [END_EVIDENCE_CHAIN]'.format(block['context'],selected_chain)
            # context = block['context']
            searched_context.append(context)
        all_context = [{'passage':c,'type':'searched'} for c in searched_context]
        # all_context = all_context[:topk_tbs]
        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        input_context_str = '[TB] '+' [TB] '.join(ranked_context)

#         all_start_pos = [orig_answer.start() for orig_answer in re.finditer(orig_answer.lower(), context_str.lower())]
#         context_str = ' [TB] '.join()
#         start = input_context_str.lower().find(orig_answer.lower())
        #randomly select a answer if there exists multiple
#         start = random.choice(all_start_pos)
#         if start == -1:
#             import pdb
#             pdb.set_trace()
#             while input_context_str[start].lower() != orig_answer[0].lower():
#                 start -= 1
#         answer = input_context_str[start:start + len(orig_answer)]

        # answer = orig_answer
        split.append({'context': input_context_str, 'title': d['table_id'],
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': 0, 'text': orig_answer}]})

        if idx < 2:
            logger.info(split[-1])
    logger.info('no candidate rate: {}'.format(error_cnt / all_cnt)) 
    logger.info('{}/{} examples are hitted'.format(len(hit), len(data)))
    return split

def prepare_qa_data_evidence_chain_ranking_retrieved_test(data, topk_tbs = 15):
    split = []
    hit = set()
    key2idx = {}

    error_cnt,all_cnt = 0,0
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        searched_context = []
        for block in d['retrieved_tbs'][:topk_tbs]:
            all_cnt+=1
            if block['candidate_evidence_chains']:
                ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: softmax(k['score'])[1], reverse=True)
                if False: #len(ranked_ec)>=2:
                    union_ec = find_union_chain(ranked_ec, topt=2)
                    selected_chain = convert_chain_to_string(union_ec)
                else:
                    selected_chain = convert_chain_to_string(ranked_ec[0]['path'])
                # selected_chain = convert_chain_to_string(ranked_ec[0]['path'])#ranked_ec[0]['path']
            else:
                selected_chain = 'none'
                error_cnt+=1
            context = '{} [EVIDENCE_CHAIN] {} [END_EVIDENCE_CHAIN]'.format(block['context'],selected_chain)
            # context = block['context']
            searched_context.append(context)
        all_context = [{'passage':c,'type':'searched'} for c in searched_context]
        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        input_context_str = '[TB] '+' [TB] '.join(ranked_context)

        split.append({'context': input_context_str, 'title': '',
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': 0, 'text': 'none'}]})
        key2idx[d['question_id']] = idx
        if idx < 2:
            logger.info(split[-1])
    logger.info('no candidate rate: {}'.format(error_cnt / all_cnt))
    return split,key2idx
def prepare_qa_data_evidence_chain_ranking(data, topk_tbs = 15):
    split = []
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        positive_table_block = d['positive_table_blocks']
        # positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        ground_truth_context = []
        gt_tb_ids = {}

        for block in positive_table_block:
            best_score = -999
            selected_chain = []
            ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: softmax(k['score'])[1], reverse=True)
            # for chain in block['candidate_evidence_chains']:
            #     positive_chain = convert_chain_to_string(chain['path'])
            #     score = softmax(chain['score'])#fuzz.partial_ratio(d['question']+' '+orig_answer,positive_chain)
            #     if score[1]>best_score:
            #         best_score = score[1]
            #         selected_chain = positive_chain
            # print(d['question'],orig_answer)
            # print(selected_chain,best_score)
            # input()
            if ranked_ec:
                for i in range(min(len(ranked_ec),1)):
                    selected_chain.append(convert_chain_to_string(ranked_ec[i]['path']))
            context = '{} [EVIDENCE_CHAIN] {} [END_EVIDENCE_CHAIN]'.format(block['context'],' [SEP] '.join(selected_chain))

            gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])] = context
            ground_truth_context.append(context)

        all_context = [{'passage':c,'type':'ground-truth'} for c in ground_truth_context]# + [{'passage':c,'type':'searched'} for c in searched_context]

        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        input_context_str = '[TB] '+' [TB] '.join(ranked_context)

#         all_start_pos = [orig_answer.start() for orig_answer in re.finditer(orig_answer.lower(), context_str.lower())]
#         context_str = ' [TB] '.join()
        start = input_context_str.lower().find(orig_answer.lower())
        #randomly select a answer if there exists multiple
#         start = random.choice(all_start_pos)
        if start == -1:
            import pdb
            pdb.set_trace()
            while input_context_str[start].lower() != orig_answer[0].lower():
                start -= 1
        answer = input_context_str[start:start + len(orig_answer)]

        # answer = orig_answer
        split.append({'context': input_context_str, 'title': d['table_id'],
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': start, 'text': answer}]})

        if idx < 2:
            logger.info(split[-1])
    # logger.info('{}/{} examples are hitted'.format(len(hit), len(data)))
    return split


from preprocessing.utils_preprocess import rank_doc_tfidf
def prepare_qa_data_tbconcat(data, topk_tbs = 15):
    split = []
    # assert len(data) == len(retrieval_outputs),'length of data {}, length of retrieval outputs {}'.format(len(data),len(retrieval_outputs))
    # TOPN = [item for item in list(retrieval_outputs[0].keys()) if 'top_' in item][0]

    hit = set()
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        positive_table_block = d['positive_table_blocks']
        # positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        ground_truth_context = []
        gt_tb_ids = {}

        for block in positive_table_block:
            context = block['context']
            # table_segment = pd.DataFrame(data=[table[block['row_id']]], columns=header)
            # # table_segment = pd.DataFrame.from_dict(block['table_segment'])
            # # passage = random.choice(block['passages']) if block['passages'] else ''
            # context = convert_tb_to_string_metadata(table_segment, block['passages'], gt_meta_data,cut='passage')
            gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])] = context
            ground_truth_context.append(context)
        searched_context = []
        for block in d['retrieved_tbs']:
            if '{}-{}'.format(block['table_id'],block['row_id']) in gt_tb_ids.keys():
                hit.add(d['question_id'])
                continue
            searched_context.append(block['context'])
#         print(ground_truth_context)
#         print(searched_context)
#         input()
        # find context based on retrieval outputs

        all_context = ground_truth_context + searched_context
        # all_context = searched_context
        all_context = all_context[:topk_tbs]
        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        #whether to shuffle the context to avoid bias
        # random.shuffle(all_context)
        input_context_str = ' [TB] '.join(ranked_context)
#         all_start_pos = [orig_answer.start() for orig_answer in re.finditer(orig_answer.lower(), context_str.lower())]
#         context_str = ' [TB] '.join()
        start = input_context_str.lower().find(orig_answer.lower())
        #randomly select a answer if there exists multiple
#         start = random.choice(all_start_pos)
        if start == -1:
            import pdb
            pdb.set_trace()
            while input_context_str[start].lower() != orig_answer[0].lower():
                start -= 1
        answer = input_context_str[start:start + len(orig_answer)]

        # answer = orig_answer
        split.append({'context': input_context_str, 'title': d['table_id'],
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': start, 'text': answer}]})
        if idx<2:
            print(split[-1])
    print('{}/{} examples are hitted'.format(len(hit),len(data)))
    return split
def prepare_qa_data_tbconcat_eval(data, topk_tbs = 15):
    split = []
    # assert len(data) == len(retrieval_outputs),'length of data {}, length of retrieval outputs {}'.format(len(data),len(retrieval_outputs))
    # TOPN = [item for item in list(retrieval_outputs[0].keys()) if 'top_' in item][0]

    hit = set()
    output_compare = []
    for idx,d in tqdm(enumerate(data), desc='Preprocessing data'):
        positive_table_block = d['positive_table_blocks']
        # positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        ground_truth_context = []
        gt_tb_ids = {}

        for block in positive_table_block:
            context = block['context']
            # table_segment = pd.DataFrame(data=[table[block['row_id']]], columns=header)
            # # table_segment = pd.DataFrame.from_dict(block['table_segment'])
            # # passage = random.choice(block['passages']) if block['passages'] else ''
            # context = convert_tb_to_string_metadata(table_segment, block['passages'], gt_meta_data,cut='passage')
            gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])] = {'context':context,'passages':block['passages']}
            ground_truth_context.append(context)
        searched_context = []
        for block in d['retrieved_tbs']:
            if '{}-{}'.format(block['table_id'],block['row_id']) in gt_tb_ids.keys():
                hit.add(d['question_id'])
                # gt_context = gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])]['context']
                # searched_context.append(gt_context)
                # output_compare.append({'searched':block['context'],'ground_truth':gt_context,'answer':orig_answer,'id':'{}-{}'.format(block['table_id'],block['row_id']),
                #                        'find_passages':block['s_psg'],'gt_passages':gt_tb_ids['{}-{}'.format(block['table_id'],block['row_id'])]['passages']})
                # continue
            searched_context.append(block['context'])
#         print(ground_truth_context)
#         print(searched_context)
#         input()
        # find context based on retrieval outputs

        all_context = searched_context
        # all_context = searched_context
        all_context = all_context[:topk_tbs]
        # random.shuffle(all_context)
        #rank the context based on tfidf
        ranked_context = [doc['passage'] for doc in rank_doc_tfidf(d['question'], all_context)]
        input_context_str = ' [TB] '.join(ranked_context)
        #whether to shuffle the context to avoid bias
        # random.shuffle(all_context)
        # input_context_str = ' [TB] '.join(all_context)

#         all_start_pos = [orig_answer.start() for orig_answer in re.finditer(orig_answer.lower(), context_str.lower())]
#         context_str = ' [TB] '.join()
#         start = input_context_str.lower().find(orig_answer.lower())
        #randomly select a answer if there exists multiple
#         start = random.choice(all_start_pos)
#         if start == -1:
#             import pdb
#             pdb.set_trace()
#             while input_context_str[start].lower() != orig_answer[0].lower():
#                 start -= 1
#         answer = input_context_str[start:start + len(orig_answer)]

        # answer = orig_answer
        split.append({'context': input_context_str, 'title': d['table_id'],
                      'question': d['question'], 'question_id': d['question_id'],
                      'answers': [{'answer_start': 0, 'text': orig_answer}]})
        if idx<2:
            print(split[-1])
    print('{}/{} examples are hitted'.format(len(hit),len(data)))

    return split
def prepare_qa_data(data, max_context_length=400):
    '''
        tokenizer-free, 后面要改输入输出就改这里就行了

    Input file format
    [{
        'question_id': str
        'question': str
        'answer-text': answer-text
        'answer-nodes': []
        'positive_table_blocks‘: {'table_id','row_id','table_segment':,'passages':[passage text]}
        'passages': [{'index':str , 'position':[row_id,col_id], 'passage':str}]
        'table': table
     }, {}]

    Output data format
    [{
        'context': str,
        'title': str,
        'question': str,
        'question_id': str,
        'answers': [{ 'answer_start': int,
                      'text': int
                      }]
     }, {}]
    '''
    split = []
    from_both, from_passage, from_cell = 0, 0, 0
    all_question_ids = Counter()
    for d in tqdm(data, desc='   Preparing QA data, find answer span: '):
        positive_table_block = d['positive_table_blocks']
        positive_passages = d['passages']
        orig_answer = d['answer-text']
        answer_node = d['answer-node']
        table = [[cell[0] for cell in row] for row in d['table']['data']]
        header = [hea[0] for hea in d['table']['header']]
        if d['where'] == 'table':
            for block in positive_table_block:
                table_segment = pd.DataFrame(data=[table[block['row_id']]], columns=header)
                # table_segment = pd.DataFrame.from_dict(block['table_segment'])
                passage = random.choice(block['passages']) if block['passages'] else ''

                context = convert_tb_to_string(table_segment, [passage], cut='passage')
                start = context.lower().find(orig_answer.lower())
                if start == -1:
                    import pdb
                    pdb.set_trace()
                    while context[start].lower() != orig_answer[0].lower():
                        start -= 1
                answer = context[start:start + len(orig_answer)]
                all_question_ids[d['question_id']] += 1
                question_id = d['question_id'] + '-{}'.format(all_question_ids[d['question_id']])
                # question_id = d['question_id']
                split.append({'context': context, 'title': d['table_id'],
                              'question': d['question'], 'question_id': question_id,
                              'answers': [{'answer_start': start, 'text': answer}]})
        else:
            for passage in positive_passages:
                block = [tb for tb in positive_table_block if int(tb['row_id']) == int(passage['position'][0])][0]
                table_segment = pd.DataFrame.from_dict(block['table_segment'])
                passage = passage['passage']
                context = convert_tb_to_string(table_segment, [passage], cut='table')
                start = context.lower().find(orig_answer.lower())
                if start == -1:
                    import pdb
                    pdb.set_trace()
                    while context[start].lower() != orig_answer[0].lower():
                        start -= 1
                answer = context[start:start + len(orig_answer)]
                all_question_ids[d['question_id']] += 1
                question_id = d['question_id'] + '-{}'.format(all_question_ids[d['question_id']])
                # question_id = d['question_id']
                split.append({'context': context, 'title': d['table_id'],
                              'question': d['question'], 'question_id': question_id,
                              'answers': [{'answer_start': start, 'text': answer}]})

    return split


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, test=True):
    # Load data features from cache or dataset file
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            args.prefix
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s to %s", input_dir, cached_features_file)

        processor = OTTQAProcessorConcat(args)
        logger.info("using the processor: {}".format(processor.__class__.__name__))
        if evaluate:
            examples = processor.get_dev_examples(args.dev_file)
        elif test:
            examples,key2idx = processor.get_test_examples()
        else:
            examples = processor.get_train_examples(args.train_file)

        features, dataset = ottqa_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer, 
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=(not evaluate) and (not test),
            threads=args.threads,
            padding_strategy="max_length"
        )
        if args.local_rank in [-1, 0] and args.save_cache:
           logger.info("Saving features into cached file %s", cached_features_file)
           torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    if output_examples:
        if test:
            return dataset, examples, features,key2idx
        return dataset, examples, features

    return dataset


MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


def ottqa_convert_example_to_features(
        example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )
        # encoded_dict['token_type_ids'] = encoded_dict['token_type_ids'] + [encoded_dict['token_type_ids'][-1]] * \
        #                                  (len(encoded_dict['input_ids'])-len(encoded_dict['token_type_ids']))  # 这句新加的跑bigbird用

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len
        ec_mask = generate_select_features(encoded_dict)
        encoded_dict['ec_mask'] = ec_mask
        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context
    tmp_features = []
    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if True:#(is_training and not span_is_impossible) or not is_training:
            tmp_features.append(
                OTTQAFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=0,
                    # Can not set unique_id and example_index here. They will be set after multiple processing.
                    unique_id=0,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    ec_mask=span['ec_mask'],
                    qas_id=example.qas_id,
                )
            )
    # features.append(tmp_features[0])
    features.extend(tmp_features)
    return features

def generate_select_features(encoded_dict):
    ec_token_id = tokenizer.convert_tokens_to_ids(['[EVIDENCE_CHAIN]'])[0]
    ec_end_token_id = tokenizer.convert_tokens_to_ids(['[END_EVIDENCE_CHAIN]'])[0]
    evidence_chain_mask,begin = [],False
    words = []
    for id,v in enumerate(encoded_dict['input_ids']):
        if v == ec_token_id:
            begin=True
        elif v==ec_end_token_id:
            begin=False
        if begin:
            evidence_chain_mask.append(1)
            #words.append(encoded_dict['tokens'][id])
        else:
            evidence_chain_mask.append(0)
    #print(words)
    #print(evidence_chain_mask)
    #exit()
    return evidence_chain_mask


def ottqa_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def ottqa_convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
                                       is_training, threads=1, padding_strategy="max_length"):
    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=ottqa_convert_example_to_features_init, initargs=(tokenizer, )) as p:
        annotate_ = partial(
            ottqa_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
            padding_strategy=padding_strategy
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="   Convert ottqa examples to index features",
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="   Add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_ec_mask = torch.tensor([f.ec_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    if not is_training:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_ec_mask,all_cls_index, all_p_mask
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_ec_mask,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
        )
    print('length of dataset: {}'.format(dataset))

    return features, dataset
