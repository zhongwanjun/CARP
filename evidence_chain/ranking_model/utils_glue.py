# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos_text_b=None, neg_text_b=None,label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.pos_text_b = pos_text_b
        self.neg_text_b = neg_text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, a_input_ids, a_input_mask, a_segment_ids,
                 pos_b_input_ids,pos_b_input_mask,pos_b_segment_ids,
                 neg_b_input_ids,neg_b_input_mask,neg_b_segment_ids,
                 label_id):
        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.pos_b_input_ids = pos_b_input_ids
        self.pos_b_input_mask = pos_b_input_mask
        self.pos_b_segment_ids = pos_b_segment_ids
        self.neg_b_input_ids = neg_b_input_ids
        self.neg_b_input_mask = neg_b_input_mask
        self.neg_b_segment_ids = neg_b_segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


#     def _read_tsv(cls, input_file, quotechar=None):
#         """Reads a tab separated value file."""
#         with open(input_file, "r", encoding="utf-8-sig") as f:
#             reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#             lines = []
#             for line in reader:
#                 if sys.version_info[0] == 2:
#                     line = list(unicode(cell, 'utf-8') for cell in line)
#                 lines.append(line)
#             return lines
import random
import copy
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
        chain_rep.append(prefix + node['content'])
    # print(' [TO] '.join(chain_rep))
    # input()
    return ' [TO] '.join(chain_rep)
def convert_chain_to_examples(chain,neg_cells,neg_passages):
    if 'id' in neg_passages[0].keys():
        passage_context = ['[PASSAGE] {} : {}'.format(neg_passages[0]['id'].replace('/wiki/', '').replace('_', ' '),neg_passages[0]['text'])] + \
                          ['[PASSAGE] {} : {}'.format(neg_passages[1]['id'].replace('/wiki/', '').replace('_', ' '),
                                                      neg_passages[1]['text'])]
    else:
        passage_context = [[]]*2
    neg_cells = ['[TAB] {} : {}'.format(chain[1]['header'],neg_cells[0]),'[TAB] {} : {}'.format(chain[2]['header'],neg_cells[1])]
    neg_contents = [passage_context[0]] + [neg_cells[0]]  + [neg_cells[1]] + [passage_context[1]]
    chain_rep = []
    all_chain_instances = []
    for node in chain:
        if 'passage' in node.keys():
            rep = '[PASSAGE] {} : {}'.format(node['passage_index'].replace('/wiki/', '').replace('_', ' '),
                node['passage'])
        else:
            rep = '[TAB] {} : {}'.format(node['header'],node['content'])
        chain_rep.append(rep)
    for nid, node in enumerate(chain):
        tmp_chain_rep = copy.deepcopy(chain_rep)
        tmp_chain_rep[nid] = '[MASK]'
        if neg_contents[nid]:
            all_chain_instances.append(
                InputExample(guid=0, text_a=' [TO] '.join(tmp_chain_rep), pos_text_b=chain_rep[nid],neg_text_b=neg_contents[nid], label='1'))
        # all_chain_instances.append(InputExample(guid=0,text_b=' [TO] '.join(tmp_chain_rep),text_a=chain_rep[nid],label='1'))
        # if neg_contents[nid]:
        #     all_chain_instances.append(InputExample(guid=0,text_b=' [TO] '.join(tmp_chain_rep), text_a=neg_contents[nid], label='0'))
    # print(' [TO] '.join(chain_rep))
    # input()
    return all_chain_instances

import json
from tqdm import tqdm


class EvidenceChainPretrainProcessor(DataProcessor):
    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self.processing_data(os.path.join(data_dir, train_file))
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self.processing_data(os.path.join(data_dir, dev_file))
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self.processing_eval_data(os.path.join(data_dir, test_file))
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # def processing_data(self, data_path):
    #     logger.info(f"Loading data from {data_path}")
    #     with open(data_path, 'r') as f:
    #         data = json.load(f) # [:100]
    #     # self.data = self.data[:300]
    #     logger.info(f"Total instances count {len(data)}")
    #     examples = []
    #     guid = 0
    #     for js in tqdm(data, desc='preparing Evidence chain pretrain dataset..'):
    #         chain = js['chain']
    #         # title = js['table_title']
    #         # sec_title = js['section_title']
    #         chain_examples = convert_chain_to_examples(chain,js['neg_cells'],js['neg_passages'])
    #         for eid in range(len(chain_examples)):
    #             chain_examples[eid].guid = guid
    #             guid += 1
    #         examples.extend(chain_examples)
    #     logger.info(f"Total instances {len(examples)}")
    #     return examples

    def processing_data(self,data_path):
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)#[:100]
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(data)}")
        examples = []
        guid = 0
        for js in tqdm(data, desc='preparing Evidence chain ranker dataset..'):
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]

            table_blocks = js['positive_table_blocks']
            for table_block in table_blocks:
                for idx,chain in enumerate(table_block['evidence_chain']['positive']):
                    positive_chain = convert_chain_to_string(chain)
                    negative_chain = convert_chain_to_string(table_block['evidence_chain']['negative'][idx])
                    examples.append(
                        InputExample(guid=guid, text_a=question, pos_text_b=positive_chain,neg_text_b=negative_chain,label='1'))
                    guid+=1
                    # examples.append(
                    #     InputExample(guid=guid, text_a=question, text_b=negative_chain, label='0'))
                    # guid+=1
        logger.info(f"Total instances {len(examples)}")
        return examples
    def processing_eval_data(self,data_path):
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)  # [:100]
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(data)}")
        examples = []
        guid = 0
        for js in tqdm(data, desc='preparing evidence chain dataset for ranking candidate paths..'):
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_blocks = js['positive_table_blocks']
            for tbid,table_block in enumerate(table_blocks):
                for idx, chain in enumerate(table_block['candidate_evidence_chains']):
                    chain_str = convert_chain_to_string(chain)
                    examples.append(
                        InputExample(guid='{}#{}#{}'.format(js['question_id'],tbid,idx), text_a=question, pos_text_b=chain_str,neg_text_b=chain_str, label='1'))

        logger.info(f"Total instances {len(examples)}")
        return examples,data
    def processing_eval_data_retrieved(self,data_path):
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)  # [:100]
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(data)}")
        examples = []
        guid = 0
        for js in tqdm(data, desc='preparing evidence chain dataset for ranking candidate paths..'):
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_blocks = js['retrieved_tbs']
            for tbid,table_block in enumerate(table_blocks):
                for idx, chain in enumerate(table_block['candidate_evidence_chains']):
                    chain_str = convert_chain_to_string(chain)
                    examples.append(
                        InputExample(guid='{}#{}#{}'.format(js['question_id'],tbid,idx), text_a=question, pos_text_b=chain_str, neg_text_b=chain_str, label='1'))

        logger.info(f"Total instances {len(examples)}")
        return examples,data

def output_pred_results(instances, all_logits):
    cnt = 0
    for iid, js in tqdm(enumerate(instances), desc='Saving output results'):
        question = js['question']
        if question.endswith("?"):
            question = question[:-1]
        table_blocks = js['positive_table_blocks']
        for tbid, table_block in enumerate(table_blocks):
            for idx, chain in enumerate(table_block['candidate_evidence_chains']):
                instances[iid]['positive_table_blocks'][tbid]['candidate_evidence_chains'][idx] ={'score': all_logits[cnt],'path':instances[iid]['positive_table_blocks'][tbid]['candidate_evidence_chains'][idx]}
                cnt+=1
    return instances
def output_pred_results_retrived(instances, all_logits):
    cnt = 0
    for iid, js in tqdm(enumerate(instances), desc='Saving output results'):
        question = js['question']
        if question.endswith("?"):
            question = question[:-1]
        table_blocks = js['retrieved_tbs']
        for tbid, table_block in enumerate(table_blocks):
            for idx, chain in enumerate(table_block['candidate_evidence_chains']):
                instances[iid]['retrieved_tbs'][tbid]['candidate_evidence_chains'][idx] ={'score': all_logits[cnt],'path':instances[iid]['retrieved_tbs'][tbid]['candidate_evidence_chains'][idx]}
                cnt+=1
    return instances
def generate_select_features(input_ids):
    mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    # print(encoded_dict['tokens'])
    mask_pos = 0
    for id,v in enumerate(input_ids):
        if v == mask_token_id:
            mask_pos = id
    return mask_pos

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,max_part_length,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        a_inputs = tokenizer.encode_plus(example.text_a,max_length=max_seq_length,
                                         add_special_tokens=True,padding='max_length',
                                         truncation=True,return_token_type_ids=True)
        pos_b_inputs = tokenizer.encode_plus(example.pos_text_b,max_length=max_part_length,
                                         add_special_tokens=True,padding='max_length'
                                         ,truncation=True,return_token_type_ids=True)
        neg_b_inputs = tokenizer.encode_plus(example.neg_text_b, max_length=max_part_length,
                                             add_special_tokens=True, padding='max_length',
                                              truncation=True,return_token_type_ids=True)
        '''
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        '''
        a_input_ids = a_inputs['input_ids']
        a_input_mask = a_inputs['attention_mask']
        a_segment_id = a_inputs['token_type_ids']
        pos_b_input_ids = pos_b_inputs['input_ids']
        pos_b_input_mask = pos_b_inputs['attention_mask']
        pos_b_segment_id = pos_b_inputs['token_type_ids']
        neg_b_input_ids = neg_b_inputs['input_ids']
        neg_b_input_mask = neg_b_inputs['attention_mask']
        neg_b_segment_id = neg_b_inputs['token_type_ids']
        # mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        # print(encoded_dict['tokens'])
        # mask_idx = 0
        # for id, v in enumerate(a_input_ids):
        #     if v == mask_token_id:
        #         mask_idx = id
        # assert len(a_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        # assert(a_input_ids[mask_idx]==mask_token_id),'{} mask idx {}: \n{}'.format(mask_token_id,mask_idx,tokenizer.convert_ids_to_tokens(a_input_ids))
        # print(tokenizer.convert_ids_to_tokens([a_input_ids[mask_idx]]))
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("a tokens: %s" % " ".join(
                [str(x) for x in tokenizer.convert_ids_to_tokens(a_input_ids)]))
            logger.info("pos b tokens: %s" % " ".join(
                [str(x) for x in tokenizer.convert_ids_to_tokens(pos_b_input_ids)]))
            logger.info("neg b tokens: %s" % " ".join(
                [str(x) for x in tokenizer.convert_ids_to_tokens(neg_b_input_ids)]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in a_input_ids]))
            # logger.info("input_mask: %s" % " ".join([str(x) for x in a_input_mask]))
            # logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(a_input_ids=a_input_ids,
                          a_input_mask=a_input_mask,
                          a_segment_ids=a_segment_id,
                          pos_b_input_ids=pos_b_input_ids,
                          pos_b_input_mask=pos_b_input_mask,
                          pos_b_segment_ids=pos_b_segment_id,
                          neg_b_input_ids=neg_b_input_ids,
                          neg_b_input_mask=neg_b_input_mask,
                          neg_b_segment_ids=neg_b_segment_id,
                          # mask_idx = mask_idx,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "evidence_chain":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    'evidence_chain':EvidenceChainPretrainProcessor
}

output_modes = {
    'evidence_chain':"classification"
}

GLUE_TASKS_NUM_LABELS = {
    'evidence_chain':2
}
