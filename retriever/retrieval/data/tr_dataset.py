import json
import pickle
import os, sys
import random
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
from .data_utils import collate_tokens
import copy
import sys
sys.path.append('../')

from utils.common import convert_tb_to_string, get_passages, load_jsonl

from transformers import BatchEncoding

from functools import partial
from pathos.multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)


def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
                ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str

def convert_tb_to_string_metadata_norm(table, passages, meta_data, cut='passage', max_length=400):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] ' + \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str


def convert_tb_to_features_bert_metadata(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    return batch_outputs


def convert_tb_to_string(table, passages, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [HEADER] ' + ' [SEP] '.join(header) + ' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str

def convert_tb_to_string_norm(table, passages, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB] ' + ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header, value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str


def convert_tb_to_features_bert(passages, table, tokenizer, args):
    if table.empty:
        inputs = tokenizer.encode_plus(passages, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        table_str, passage_str = convert_tb_to_string(table, passages)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    # inputs['length'] = len(inputs['input_ids'])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    return batch_outputs



class TRDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 args,
                 train=False,
                 ):
        super().__init__()
        self.train = train
        self.question = []
        self.table_block = []
        self.labels = []

        logger.info(f"Loading data from {data_path}")
        self.data = []
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Total sample count {len(self.data)}")

        for js in tqdm(self.data, desc='preparing dataset..'):
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            self.labels.append(torch.tensor(js['label']))

    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.data)


class TRDatasetNega(Dataset):

    def __init__(self, tokenizer, data_path, args, train=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.args = args
        self.train = train
        self.question = []
        self.table_block = []
        self.negative_table_block = []
        self.labels = []
        self.data = []

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            for js in tqdm(self.data, desc='preparing singleRetriever dataset..'):
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                self.question.append(convert_tb_to_features_bert(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args))
                self.table_block.append(convert_tb_to_features_bert(js['passages'], js['table'],
                                                                    tokenizer=self.tokenizer, args=self.args))
                self.negative_table_block.append(convert_tb_to_features_bert(js['neg_passages'], js['neg_table'],
                                                                             tokenizer=self.tokenizer, args=self.args))

                self.labels.append(torch.tensor(js['label']))
            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "neg_tb": self.negative_table_block[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.question)

    def save_tensor(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.question, os.path.join(path, 'question.pkl'))
        logger.info("Saving training question \t tensors to {}".format(os.path.join(path, 'question.pkl')))
        torch.save(self.table_block, os.path.join(path, 'table_block.pkl'))
        logger.info("Saving training table_block \t tensors to {}".format(os.path.join(path, 'table_block.pkl')))
        torch.save(self.negative_table_block, os.path.join(path, 'negative_table_block.pkl'))
        logger.info("Saving training negative_table_block \t tensors to {}".format(
            os.path.join(path, 'negative_table_block.pkl')))
        torch.save(self.labels, os.path.join(path, 'labels.pkl'))
        logger.info("Saving training labels \t tensors to {}".format(os.path.join(path, 'labels.pkl')))
        logger.info("Saving training (q, tb, neg_tb, lab) tensors to {}".format(path))

    def load_tensor(self, path):
        self.question = torch.load(os.path.join(path, 'question.pkl'))
        logger.info("Loading training question \t tensors from {}".format(os.path.join(path, 'question.pkl')))
        self.table_block = torch.load(os.path.join(path, 'table_block.pkl'))
        logger.info("Loading training table_block \t tensors from {}".format(os.path.join(path, 'table_block.pkl')))
        self.negative_table_block = torch.load(os.path.join(path, 'negative_table_block.pkl'))
        logger.info("Loading training negative_table_block \t tensors from {}".format(
            os.path.join(path, 'negative_table_block.pkl')))
        self.labels = torch.load(os.path.join(path, 'labels.pkl'))
        logger.info("Loading training labels \t tensors from {}".format(os.path.join(path, 'labels.pkl')))
        logger.info("Loading training (q, tb, neg_tb, lab) tensors from {}".format(path))


class TRDatasetNegaMeta(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaMeta, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                output['question'] = convert_tb_to_features_bert_metadata(question, pd.DataFrame([]), js['meta_data'],
                                                                          tokenizer=tokenizer, args=args)
                psg = get_passages(js, args.psg_mode, neg=False)[:8]
                output['table_block'] = convert_tb_to_features_bert_metadata(psg, js['table'], js['meta_data'],
                                                                    tokenizer=tokenizer, args=args)
                psg = get_passages(js, args.psg_mode, neg=True)[:8]
                output['negative_table_block'] = convert_tb_to_features_bert_metadata(psg, js['neg_table'], js['meta_data'],
                                                                             tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output
            logger.info("normalize table text")
            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)


def collate_all_tokens(input_type, inputs, *args):
    return_dict = {}
    for arg in args:
        return_dict[input_type + '_' + arg] = collate_tokens([input[input_type][arg].view(-1) for input in inputs], 0)
    return return_dict


def tb_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'q_input_ids': collate_tokens([s["q"]["input_ids"].view(-1) for s in samples], pad_id),
        'q_mask': collate_tokens([s["q"]["attention_mask"].view(-1) for s in samples], 0),
        'c_input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'c_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
        'neg_input_ids': collate_tokens([s["neg_tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'neg_mask': collate_tokens([s["neg_tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["q"]:
        batch.update({
            'q_part2_mask': collate_tokens([s["q"]["part2_mask"].view(-1) for s in samples], 0),
            'c_part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
            'neg_part2_mask': collate_tokens([s["neg_tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["q"]:
        batch.update({
            'q_part3_mask': collate_tokens([s["q"]["part3_mask"].view(-1) for s in samples], 0),
            'c_part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
            'neg_part3_mask': collate_tokens([s["neg_tb"]["part3_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["q"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q"]["token_type_ids"] for s in samples], 0, '2d'),
            'c_type_ids': collate_tokens([s["tb"]["token_type_ids"] for s in samples], 0, '2d'),
            'neg_type_ids': collate_tokens([s["neg_tb"]["token_type_ids"] for s in samples], 0, '2d'),
        })

    return batch


