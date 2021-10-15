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
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import sys
from io import open
import random
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return json.dumps({'text_a':self.text_a,'text_b':self.text_b,'label':self.label},indent=4)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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
def convert_chain_to_examples(chain,neg_cells,neg_passages):
    if 'id' in neg_passages[0].keys():
        passage_context = ['[PASSAGE] {} : {}'.format(neg_passages[0]['id'].replace('/wiki/', '').replace('_', ' '),neg_passages[0]['text'])] + \
                          ['[PASSAGE] {} : {}'.format(neg_passages[1]['id'].replace('/wiki/', '').replace('_', ' '),
                                                      neg_passages[1]['text'])]
    else:
        passage_context = [[]]*2
    neg_contents = [neg_cells[0]] + [passage_context[0]] + [neg_cells[1]] + [passage_context[1]]
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
        # if neg_contents[nid]:
            # all_chain_instances.append(InputExample(guid))
        all_chain_instances.append(InputExample(guid=0,text_b=' [TO] '.join(tmp_chain_rep),text_a=chain_rep[nid],label='1'))
        if neg_contents[nid]:
            all_chain_instances.append(InputExample(guid=0,text_b=' [TO] '.join(tmp_chain_rep), text_a=neg_contents[nid], label='0'))
    # print(' [TO] '.join(chain_rep))
    # input()
    return all_chain_instances

import json
from tqdm import tqdm

class EvidenceChainPretrainProcessorNew(DataProcessor):
    def __init__(self):
        super(EvidenceChainPretrainProcessorNew, self).__init__()

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
        return self.processing_data(os.path.join(data_dir, test_file))
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def processing_data(self, data_path):
        logger.info(f"Loading data from {data_path}")
        f = open(data_path, 'r')
        # with open(data_path.replace('shortest_esnegs.jsonl','esnegs.jsonl'), 'r') as all_f:
        #     lines = all_f.readlines()
        #     data = [json.loads(line.strip()) for line in lines]
            # data = json.load(f)  # [:100]
        # self.data = self.data[:300]
        # logger.info(f"Total instances count {len(data)}")
        examples = []
        guid = 0
        error_cnt = 0
        for idx,js in tqdm(enumerate(f), desc='preparing Evidence chain pretrain dataset..'):
            js = json.loads(js.strip())
            chain = js['chain']
            question = js['question']
            table = js['tb']
            neg_chains = js['neg_chains']
            examples.append(
                InputExample(guid=guid, text_a='{} {}'.format(question,table), text_b=chain, label='1'))
            # print(InputExample(guid=guid, text_a='{} {}'.format(question,table), text_b=chain, label='1'))
            guid += 1
            if neg_chains:
                # select_num = min(len(neg_chains),3)
                # neg_chain = random.choices(neg_chains,k=select_num)
                neg_chain = [neg_chains[0]]
                for nc in neg_chain:
                    examples.append(
                        InputExample(guid=guid, text_a='{} {}'.format(question, table), text_b=nc, label='0'))
                    # print(InputExample(guid=guid, text_a='{} {}'.format(question, table), text_b=nc, label='0'))
                    guid+=1
            else:
                error_cnt+=1
            # input()
        logger.info(f"Total instances {len(examples)}")
        logger.info(f"{error_cnt}/{len(data)} examples don't have neg instances")
        return examples

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

    def processing_data(self, data_path):
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)  # [:100]
        # self.data = self.data[:300]
        logger.info(f"Total instances count {len(data)}")
        examples = []
        guid = 0
        for js in tqdm(data, desc='preparing Evidence chain pretrain dataset..'):
            chain = js['chain']
            # title = js['table_title']
            # sec_title = js['section_title']
            chain_examples = convert_chain_to_examples(chain,js['neg_cells'],js['neg_passages'])
            for eid in range(len(chain_examples)):
                chain_examples[eid].guid = guid
                guid += 1
            examples.extend(chain_examples)
        logger.info(f"Total instances {len(examples)}")
        return examples


class EvidenceChainProcessor(DataProcessor):
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
                        InputExample(guid=guid, text_a=question, text_b=positive_chain, label='1'))
                    guid+=1
                    examples.append(
                        InputExample(guid=guid, text_a=question, text_b=negative_chain, label='0'))
                    guid+=1
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
                        InputExample(guid='{}#{}#{}'.format(js['question_id'],tbid,idx), text_a=question, text_b=chain_str, label='1'))

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


def convert_example_to_feature_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def convert_example_to_feature(example,max_seq_length,label_map,output_mode='classification'):
    try:
        inputs = tokenizer.encode_plus(example.text_a, text_pair=example.text_b, max_length=max_seq_length,
                                       add_special_tokens=True, padding='max_length',
                                       truncation='only_first', return_token_type_ids=True)
    except Exception as e:
        print(e)
        if example.text_b:
            inputs = tokenizer.encode_plus(example.text_b, max_length=max_seq_length,
                                           add_special_tokens=True, padding='max_length',
                                           truncation='only_first', return_token_type_ids=True)
        else:
            inputs = tokenizer.encode_plus(example.text_a, max_length=max_seq_length,
                                           add_special_tokens=True, padding='max_length',
                                           truncation='only_first', return_token_type_ids=True)
    input_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    segment_ids = inputs['token_type_ids']
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_id)

def convert_examples_to_features_optimize(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    threads = 20
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_feature_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_feature,
            max_seq_length=max_seq_length,
            label_map=label_map
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="   Convert examples to index features",
            )
        )
    for i in range(5):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (str(i)))
        logger.info("tokens: %s" % " ".join(
            [str(x) for x in tokenizer.convert_ids_to_tokens(features[i].input_ids)]))
        logger.info("label: %s"%str(features[i].label_id))

    return features
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
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
        try:
            inputs = tokenizer.encode_plus(example.text_a,text_pair=example.text_b,max_length=max_seq_length,
                                             add_special_tokens=True, padding='max_length',
                                             truncation='only_first', return_token_type_ids=True)
        except Exception as e:
            print(e)
            if example.text_b:
                inputs = tokenizer.encode_plus(example.text_b, max_length=max_seq_length,
                                               add_special_tokens=True, padding='max_length',
                                               truncation='only_first', return_token_type_ids=True)
            else:
                inputs = tokenizer.encode_plus(example.text_a, max_length=max_seq_length,
                                               add_special_tokens=True, padding='max_length',
                                               truncation='only_first', return_token_type_ids=True)
        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']
        segment_ids = inputs['token_type_ids']
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
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
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
    'evidence_chain':EvidenceChainPretrainProcessorNew
}

output_modes = {
    'evidence_chain':"classification"
}

GLUE_TASKS_NUM_LABELS = {
    'evidence_chain':2
}
