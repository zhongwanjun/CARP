import datetime
import pickle
import json
from tqdm import tqdm
import numpy as np
from scipy.linalg import norm
def get_current_time_str():
    return str(datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
def load_json(filename):
    with open(filename,'r',encoding='utf8') as f:
        return json.load(f)

def find_similar_sentence_tfidf(sentence1,corpus,origin,cv,topk=5):
    if corpus:
        text_corpus = [sentence1] + corpus#[doc['passage'] for doc in corpus]
        vecs = cv.fit_transform(text_corpus).toarray()
        que_vec = vecs[0]
        scores = []
        for idx, doc_vec in enumerate(vecs[1:]):
            score = np.dot(que_vec, doc_vec) / (norm(que_vec) * norm(doc_vec))
            scores.append(score)
        scored_corpus = [(doc,score,ori) for doc,score,ori in zip(corpus,scores,origin)]
        results = sorted(scored_corpus,key=lambda k:k[1],reverse=True)
        topk = min(len(results),topk)
        return [res[2] for res in results[:topk]]
        # for res in results:
        #     if res[0] not in sentence1:
        #         # print(sentence1,' ; ',res[0])
        #         return res[0]
        # return results[0][0]
    else:
        return None
def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)
    return d_list

def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def save_json(results,filename):
    with open(filename,'w',encoding='utf8') as inf:
        json.dump(results,inf)

def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
    #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title'] + ' [DATA] '+\
                ' ; '.join(['{} is {}'.format(h,c) for h,c in zip(header,value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)

    return '{} {}'.format(table_str, passage_str)
def convert_tb_to_string_metadata_old(table, passages, meta_data, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
                ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return '{} {}'.format(table_str, passage_str)

def convert_tb_to_string(table, passages, cut='passage', max_length=460, topk_block=15):
    header = table.columns.tolist()
    value = table.values.tolist()
    # table_str = '[HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
    table_str = '[HEADER] ' + ' '.join(['{} is {}'.format(h, c) for h, c in zip(header, value[0])])
    # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
    #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    if cut == 'passage':
        table_length = min(max_length, len(table_str.split(' ')))
        doc_length = 0 if table_length >= max_length else max_length - table_length
    else:
        doc_length = min(max_length, len(passage_str.split(' ')))
        table_length = 0 if doc_length >= max_length else max_length - doc_length

    # table_str = ' '.join(table_str.split(' ')[:table_length])
    # passage_str = ' '.join(passage_str.split(' ')[:doc_length])
    return '{} {}'.format(table_str, passage_str)

def convert_table_to_string(table, meta_data=None, max_length=90):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = '[HEADER] ' + ' '.join(['{} is {}'.format(h, c) for h, c in zip(header, value[0])])
    # table_str = '[HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
    if meta_data:
        table_str = '[TAB] [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + ' ' + table_str
    # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
    #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
    return table_str

# def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
#                 ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
#     passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
#     return '{} {}'.format(table_str, passage_str)
#
# def convert_tb_to_string(table, passages, cut='passage', max_length=460, topk_block=15):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = '[HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
#     # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
#     #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
#     passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
#     if cut == 'passage':
#         table_length = min(max_length, len(table_str.split(' ')))
#         doc_length = 0 if table_length >= max_length else max_length - table_length
#     else:
#         doc_length = min(max_length, len(passage_str.split(' ')))
#         table_length = 0 if doc_length >= max_length else max_length - doc_length
#
#     # table_str = ' '.join(table_str.split(' ')[:table_length])
#     # passage_str = ' '.join(passage_str.split(' ')[:doc_length])
#     return '{} {}'.format(table_str, passage_str)
#
# def convert_table_to_string(table, meta_data=None, max_length=90):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = '[HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
#     if meta_data:
#         table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + table_str
#     # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
#     #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] ' + ' [SEP] '.join(value[0])
#     return table_str



def get_passages(js, psg_mode, neg=False):
    prefix = "neg_" if neg else ""
    if psg_mode=='ori':
        psg = js[prefix+"passages"]
    elif psg_mode=="s_sent":
        psg = js[prefix+'s_sent'] if len(js[prefix+'s_sent']) > 0 else js[prefix+"passages"]
    elif psg_mode=="s_psg":
        psg = js[prefix+'s_psg'] if len(js[prefix+'s_psg']) > 0 else js[prefix+"passages"]
    else:
        psg = []
    return psg

