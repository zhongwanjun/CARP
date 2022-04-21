import json
import random
from itertools import combinations
from fuzzywuzzy import fuzz
from tqdm import tqdm
import nltk
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
cv = TfidfVectorizer(tokenizer=lambda s: s.split())
def read_jsonl(path):
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            data.append(json.loads(line))
    return data

def find_most_similar(query,cands):
    best, b_score = cands[0], 0
    for idx, cand in enumerate(cands):
        if isinstance(cand,dict):
            score = fuzz.ratio(query, cand['clean_id'])
        else:
            score = fuzz.ratio(query, cand)
        if score > b_score:
            b_score = score
            best = cand
    return best

def find_most_similar_neg_psg(psg_index, neg_candidates):
    def clean(psg_index, cands):
        pidx = psg_index.replace('/wiki/','').replace('_',' ')
        for idx,item in enumerate(cands):
            neg_candidates[idx]['clean_id'] = item['id'].replace('/wiki/','').replace('_',' ')
        # tmp_cands = [item['id'].replace('/wiki/','').replace('_',' ') for item in cands]
        return pidx
    clean_pid = clean(psg_index,neg_candidates)
    best_neg = find_most_similar(clean_pid,neg_candidates)
    return best_neg

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)

    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def convert_tb_to_string_metadata(header, values, passages, meta_data, cut='passage', max_length=400):
    # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
    #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + ' [DATA] ' \
                                                                                                             ' ; '.join(
        ['{} is {}'.format(h[0], c[0]) for h, c in zip(header, values)])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return '{} {}'.format(table_str, passage_str)
from fuzzywuzzy import fuzz
def find_similar_sentence(sentence1,corpus):
    sentences = [{'sentence': sen, 'score': fuzz.token_sort_ratio(sentence1, sen)} for sen in corpus]
    sentences = sorted(sentences, key=lambda k: k['score'], reverse=True)
    for sen in sentences:
        if sen['sentence']!=sentence1:
            return sen['sentence']

def find_similar_sentence_tfidf(sentence1,corpus):
    if corpus:
        text_corpus = [sentence1] + corpus#[doc['passage'] for doc in corpus]
        vecs = cv.fit_transform(text_corpus).toarray()
        que_vec = vecs[0]
        scores = []
        for idx, doc_vec in enumerate(vecs[1:]):
            score = np.dot(que_vec, doc_vec) / (norm(que_vec) * norm(doc_vec))
            scores.append(score)
        scored_corpus = [(doc,score) for doc,score in zip(corpus,scores)]
        results = sorted(scored_corpus,key=lambda k:k[1],reverse=True)
        # for res in results:
        #     if res[0] not in sentence1:
        #         # print(sentence1,' ; ',res[0])
        #         return res[0]
        num = min(len(results),5)
        similar_sens = [sen[i][0] for i in range(num)]
        return similar_sens
    else:
        return None
#
# def find_similar_sentence_tfidf(sentence1,corpus):
#     tokenized_corpus = []
#     keys = []
#     for cid, sen in corpus:
#         tokenized_corpus.append(tokenizer.tokenize(sen))
#         keys.append(cid)
#     try:
#         corpus_feature = tfidf.fit_transform(tokenized_corpus)
#         transformed = True
#     except Exception as e:
#         # logger.info("only containing stop words, skip it")
#         print(e)
#         transformed = False
#     if transformed:
#         q_feature = tfidf.transform(nltk.word_tokenize(sentence1))
#         para_tfidf_dist = pairwise_distances(q_feature, corpus_feature, 'cosine')[0]
#         sorted_passages = [(k, sen, d) for k, sen, d in zip(keys, corpus, para_tfidf_dist)]
#         sorted_passages = sorted(sorted_passages, key=lambda x: x[2], reverse=False)  # 升序
#         return sorted_passages[0][1]
#     else:
#         return [(k, para, 1) for k, para in zip(keys, corpus)][0][1]

def select_sentence(pindex, passage):
    sentences = nltk.sent_tokenize(passage)
    clean_pindex = pindex.replace('/wiki/','').replace('_',' ')
    sentences = [{'sentence':sen,'score':fuzz.token_sort_ratio(clean_pindex,sen)} for sen in sentences]
    sentences = sorted(sentences,key=lambda k: k['score'],reverse=True)
    num = min(len(sentences),3)
    return [sentences[sid]['sentence'] for sid in range(num)]

def convert_chain_to_string():
    chain_rep = []
    for node in chain[1:]:
        if node['origin']['where'] == 'table':
            prefix = '[TAB] '
        elif node['origin']['where'] == 'passage':
            prefix = '[PASSAGE] {} : '.format(node['origin']['index'].replace('/wiki/', '').replace('_', ' ').split('/')[0])
        else:
            prefix = '[QUESTION] '
        chain_rep.append(prefix + node['content'])
    return ' ; '.join(chain_rep)

def process_wiki_tables_with_hyphen(all_passages):
    print("all passages number: {}".format(len(all_passages)))
    hephen_keys = [item for item in all_passages.keys() if '-' in item]
    hephen_keys_convert = [key.replace('-', '_') for key in hephen_keys]
    print("length of keys in all passages with hyphen:{}".format(len(all_passages)))
    added_passages = {k2: all_passages[k1] for k1, k2 in zip(hephen_keys, hephen_keys_convert)}
    print("length of added_passages with replaced hyphen:{}".format(len(added_passages)))
    all_passages.update(added_passages)
    print("new all passages number: {}".format(len(all_passages)))
    return all_passages
def generate_chains(seq):
    try:
        select_s1 = select_sentence(seq[0]['passage_index'], seq[0]['passage'])[0]
        select_s2 = select_sentence(seq[3]['passage_index'], seq[3]['passage'])[0]
    except Exception as e:
        print(e)
        return []
    s1_rep = '[PASSAGE] {} : {}'.format(seq[0]['passage_index'].replace('/wiki/', '').replace('_', ' '),select_s1)
    s2_rep = '[PASSAGE] {} : {}'.format(seq[3]['passage_index'].replace('/wiki/', '').replace('_', ' '), select_s2)
    cell1_rep = '[TAB] {} is {}'.format(seq[1]['header'],seq[1]['content'])
    cell2_rep = '[TAB] {} is {}'.format(seq[2]['header'],seq[2]['content'])
    hop1 = [[s1_rep],[s2_rep],[cell1_rep],[cell2_rep]]
    hop2 = [[s1_rep,cell1_rep],[cell1_rep,s1_rep],[cell2_rep,s2_rep],[s2_rep,cell2_rep]]
    hop3 = [[s1_rep,cell1_rep,cell2_rep],[s2_rep,cell2_rep,cell1_rep]]
    hop4 = [[s1_rep,cell1_rep,cell2_rep,s2_rep],[s2_rep,cell2_rep,cell1_rep,s1_rep]]
    all_hops = hop1+hop2+hop3+hop4
    all_hops_rep = [' [TO] '.join(chain) for chain in all_hops]
    all_hops_rep = random.choices(all_hops_rep,weights=[0.05]*len(hop1)+[0.2]*len(hop2)+[0.25]*len(hop3)+[0.4]*len(hop4),k=4)
    return all_hops_rep
def generate_negative_chains(seq):
    output_hops_rep = []
    try:
        select_all_s1 = select_sentence(seq[0]['passage_index'], seq[0]['passage'])
        select_all_s2 = select_sentence(seq[3]['passage_index'], seq[3]['passage'])
    except Exception as e:
        print(e)
        return []
    for select_s1 in select_all_s1:
        for select_s2 in select_all_s2:
            s1_rep = '[PASSAGE] {} : {}'.format(seq[0]['passage_index'].replace('/wiki/', '').replace('_', ' '),select_s1)
            s2_rep = '[PASSAGE] {} : {}'.format(seq[3]['passage_index'].replace('/wiki/', '').replace('_', ' '), select_s2)
            cell1_rep = '[TAB] {} is {}'.format(seq[1]['header'],seq[1]['content'])
            cell2_rep = '[TAB] {} is {}'.format(seq[2]['header'],seq[2]['content'])
            hop1 = [[s1_rep],[s2_rep],[cell1_rep],[cell2_rep]]
            hop2 = [[s1_rep,cell1_rep],[cell1_rep,s1_rep],[cell2_rep,s2_rep],[s2_rep,cell2_rep]]
            hop3 = [[s1_rep,cell1_rep,cell2_rep],[s2_rep,cell2_rep,cell1_rep]]
            hop4 = [[s1_rep,cell1_rep,cell2_rep,s2_rep],[s2_rep,cell2_rep,cell1_rep,s1_rep]]
            all_hops = hop1+hop2+hop3+hop4
            all_hops_rep = [' [TO] '.join(chain) for chain in all_hops]
            all_hops_rep = random.choices(all_hops_rep,weights=[0.05]*len(hop1)+[0.2]*len(hop2)+[0.25]*len(hop3)+[0.4]*len(hop4),k=2)
            output_hops_rep.extend(all_hops_rep)
    return output_hops_rep
def clean_index(index):
    return index.replace('/wiki/', '').replace('_', ' ')
def generate_pretrain_data(seq,title,table):
    try:
        select_s1 = select_sentence(seq[0]['passage_index'], seq[0]['passage'])
        select_s2 = select_sentence(seq[3]['passage_index'], seq[3]['passage'])
    except Exception as e:
        print(e)
        return []
    s1_rep = '[PASSAGE] {}'.format(select_s1)
    s2_rep = '[PASSAGE] {}'.format(select_s2)
    cell1_rep = '[TAB] {} is {}'.format(seq[1]['header'],seq[1]['content'])
    cell2_rep = '[TAB] {} is {}'.format(seq[2]['header'],seq[2]['content'])
    hop1 = [[s1_rep],[s2_rep],[cell1_rep],[cell2_rep]]
    hop1_q = [[title,seq[0]['passage_index'].replace('/wiki/', '').replace('_', ' ')],
              [title,seq[3]['passage_index'].replace('/wiki/', '').replace('_', ' ')],
              [title,seq[1]['header']],
              [title,seq[2]['header']]]

    hop2 = [[s1_rep,cell1_rep],[cell1_rep,s1_rep],[cell2_rep,s2_rep],[s2_rep,cell2_rep]]
    hop2_q = [[clean_index(seq[0]['passage_index']),seq[1]['header']],
              [seq[1]['header'], clean_index(seq[0]['passage_index'])],
              [seq[2]['header'], clean_index(seq[3]['passage_index'])],
              [clean_index(seq[3]['passage_index']),seq[2]['header']]]
    hop3 = [[s1_rep,cell1_rep,cell2_rep],[s2_rep,cell2_rep,cell1_rep]]
    hop3_q = [[clean_index(seq[0]['passage_index']),  seq[2]['header']]
              ,[clean_index(seq[3]['passage_index']), seq[1]['header']]]
    hop4 = [[s1_rep,cell1_rep,cell2_rep,s2_rep],[s2_rep,cell2_rep,cell1_rep,s1_rep]]
    hop4_q = [[clean_index(seq[0]['passage_index']), seq[2]['header'],clean_index(seq[3]['passage_index'])],
              [clean_index(seq[3]['passage_index']), seq[1]['header'],clean_index(seq[0]['passage_index'])]]
    all_questions = []
    chains = hop2 + hop3 + hop4
    for qid,q in enumerate(hop2_q+hop3_q+hop4_q):
        qlen = len(q)
        pos = random.randint(0,qlen+1)
        new_q = q[0:pos]+[title]+q[pos:]
        all_questions.append({'question':' '.join(new_q)+' ?','chain':' [TO] '.join(chains[qid])})
    hop1_questions = [{'question':' '.join(hop1_q[idx]),'chain':' [TO] '.join(hop1[idx])} for idx in range(len(hop1))]
    all_questions = hop1_questions + all_questions
    # all_hops_rep = [' [TO] '.join(chain) for chain in all_hops]
    selected_question = random.choices(all_questions, weights=[0.1] * len(hop1) + [0.2] * len(hop2) + [0.25] * len(hop3) + [0.4] * len(hop4),
                   k=2)
    # for qid in range(len(selected_question)):
    #     # neg_chain = find_similar_sentence(sentence1=selected_question[qid]['chain'],corpus=all_hops_rep)
    #     # selected_question[qid]['neg_chain'] = neg_chain
    #     selected_question[qid]['table'] = table
    if isinstance(selected_question,dict):
        return [selected_question]
    return selected_question
def generate_negative_candidates(two_hop_psg_pairs,tid,header,table_row):
    negative_instances = []
    for pair in two_hop_psg_pairs:
        # print(pair)
        cid1, cid2 = int(pair[0][0].split(',')[1]), int(pair[1][0].split(',')[1])
        if pair[0][1][0] in all_passages.keys() and pair[1][1][0] in all_passages.keys():
            seq = [{'passage_index': pair[0][1][0], 'passage': all_passages[pair[0][1][0]]},
                   {'table_cell': '{}/{}'.format(tid, pair[0][0]), 'header': header[cid1][0],
                    'content': table_row[cid1][0]},
                   {'table_cell': '{}/{}'.format(tid, pair[1][0]), 'header': header[cid2][0],
                    'content': table_row[cid2][0]},
                   {'passage_index': pair[1][1][0], 'passage': all_passages[pair[1][1][0]]}]
            pretrain_chain_reps = generate_chains(seq)
            negative_instances.extend(pretrain_chain_reps)

    return negative_instances
def find_negatives(pretrain_chain_reps,neg_candidates):
    negatives = []
    for chain_rep in pretrain_chain_reps:
        tmp_negs = find_similar_sentence_tfidf(chain_rep,neg_candidates)
        if tmp_negs:
            negatives.append(tmp_negs)
        else:
            negatives.append([])

def process(table,ttype='fake_pretrain_data'):
    # for tid, table in tqdm(list(all_tables.items())[int(length/2):],desc='Processing tables'):
    tid, table = table
    url = table['url']
    tb2negs={}
    title = table['title']
    header = [h[0] for h in table['header']]
    content = table['data']
    ori_contents = [[cell[0] for cell in row] for row in table['data']]
    mapping_entity = {}
    position2wiki = []
    all_pretrain_instances = []
    cnt,error_cnt=0,0
    non_found = set()

    for row_idx, row in enumerate(table['data']):
        position2wiki_i = {}
        row_pretrain_instances = []
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[1]):
                if ent:
                    ent = ent.replace('-','_')
                    mapping_entity[ent] = mapping_entity.get(ent, []) + [(row_idx, col_idx)]
                    position2wiki_i[f'{row_idx},{col_idx}'] = position2wiki_i.get(f'{row_idx},{col_idx}', []) + [ent]
        position2wiki.append(position2wiki_i)
        pos_passages_index = [item for k, v in list(position2wiki_i.items()) for item in v]
        pos_passages = []
        for pindex in pos_passages_index:
            if pindex in all_passages.keys():
                pos_passages.append(all_passages[pindex])
        tb_rep =  convert_tb_to_string_metadata(header,ori_contents[row_idx],pos_passages,{'title':title,'section_title':table['section_title']})
        # try:
        #     top3_neg_psgs = [item for items in tbid2docs['{}_{}'.format(tid, row_idx)] for item in items if
        #                  item['id'] not in pos_passages_index]
        # except Exception as e:
        #     print(e)
        #     top3_neg_psgs = []

        if len(position2wiki_i) >= 2:
            two_hop_psg_pairs = list(combinations(list(position2wiki_i.items()), 2))#random.choices(list(position2wiki_i.items()),2)
            if ttype=='pretrain_negatives':
                all_neg_candidates = generate_negative_candidates(two_hop_psg_pairs,tid,header,ori_contents[row_idx])
                tb2negs[tb_rep] = all_neg_candidates
                continue
            for pair_id, pair in enumerate(two_hop_psg_pairs):
                # print(pair)
                cid1,cid2 = int(pair[0][0].split(',')[1]), int(pair[1][0].split(',')[1])
                # neg_candidates = [all_neg_candidates[i] for i in range(len(all_neg_candidates)) if i != pair_id]
                cnt+=1
                if pair[0][1][0] in all_passages.keys() and pair[1][1][0] in all_passages.keys():
                    seq = [{'passage_index':pair[0][1][0],'passage':all_passages[pair[0][1][0]]},
                           {'table_cell':'{}/{}'.format(tid,pair[0][0]),'header':header[cid1][0],'content':ori_contents[row_idx][cid1][0]},
                           {'table_cell':'{}/{}'.format(tid,pair[1][0]),'header':header[cid2][0],'content':ori_contents[row_idx][cid2][0]},
                           {'passage_index':pair[1][1][0],'passage':all_passages[pair[1][1][0]]}]

                    if ttype == 'bart_inference_data':
                        pretrain_chain_reps = generate_chains(seq)
                        # negative_chain_reps = find_negatives(pretrain_chain_reps,neg_candidates)
                        all_pretrain_instances.extend([{'input': f'{tb_rep} [EC] {chain_rep}', 'output': '','tb':tb_rep,'ec':chain_rep} for chain_rep,negative_chain_rep in zip(pretrain_chain_reps,negative_chain_reps)])
                    elif ttype == 'fake_pretrain_data':
                        row_pretrain_instances.append(generate_pretrain_data(seq,title,tb_rep))
                    # if top3_neg_psgs:
                    #     neg_psg_1 = find_most_similar_neg_psg(pair[0][1][0],top3_neg_psgs)
                    #     neg_psg_2 = find_most_similar_neg_psg(pair[1][1][0], top3_neg_psgs)
                    #     neg_cell_1 = find_most_similar(ori_contents[row_idx][cid1],[tmp_row[cid1] for tmp_rid,tmp_row in enumerate(ori_contents) if tmp_rid!=row_idx])
                    #     neg_cell_2 = find_most_similar(ori_contents[row_idx][cid2],[tmp_row[cid2] for tmp_rid,tmp_row in enumerate(ori_contents) if tmp_rid!=row_idx])
                    # else:
                    #     neg_psg_1,neg_psg_2,neg_cell_1,neg_cell_2={'index':[],'passages':[]},{'index':[],'passages':[]},[],[]
                    #
                    # all_instances.append({'chain':seq,'table_row':row,'table_title':title,'section_title':table['section_title'],'url':url,'header':header,'row_id':row_idx,
                    #                   'neg_passages':[neg_psg_1,neg_psg_2],
                    #                   'neg_cells':[neg_cell_1,neg_cell_2]})
                else:
                    if pair[0][1][0] not in all_passages.keys():
                        non_found.add(pair[0][1][0])
                    elif pair[1][1][0] not in all_passages.keys():
                        non_found.add(pair[1][1][0])
            if ttype == 'fake_pretrain_data':
                for seq_id in range(len(row_pretrain_instances)):
                    cands = [item['chain'] for iid,items in enumerate(row_pretrain_instances) for item in items if iid!=seq_id]
                    for qid, q in enumerate(row_pretrain_instances[seq_id]):
                        neg_chain = find_similar_sentence_tfidf(q['chain'],cands)
                        if neg_chain:
                            row_pretrain_instances[seq_id][qid]['neg_chain'] = neg_chain
                            row_pretrain_instances[seq_id][qid]['table'] = tb_rep
                        else:
                            row_pretrain_instances[seq_id].pop(qid)
                all_pretrain_instances.extend([item for items in row_pretrain_instances for item in items])


    return all_pretrain_instances,non_found,tb2negs


if __name__ == '__main__':
    basic_dir = './ODQA'
    # all_tables = json.load(open(f'{basic_dir}/OTT-QA/data/traindev_tables.json','r',encoding='utf8'))
    all_tables = json.load(open(f'{basic_dir}/data/data_wikitable/all_tables.json', 'r', encoding='utf8'))
    all_passages = json.load(open(f'{basic_dir}/OTT-QA/data/all_passages.json','r',encoding='utf8'))
    all_passages = process_wiki_tables_with_hyphen(all_passages)
    # tmp_data = read_jsonl(f'{basic_dir}/OTT-QA/link_generator/tfidf_augmentation_results_addtop3.json')
    # tbid2docs = {}
    # for line in tmp_data:
    #     tbid2docs[line[0]] = line[2]
    # print('length of the table id 2 documents： {}'.format(len(tbid2docs.keys())))
    all_instances = []
    all_pretrain_instances = []
    global tbrep2negs

    length = len(list(all_tables.items()))
    print('length of all tables is {}'.format(len(all_tables)))
    print('length of all passages is {}'.format(len(all_passages)))
    n_threads = 20
    running_function = process
    type = sys.argv[1]
    # all_results = []
    # func_ = partial(running_function, ttype=type)
    # for item in list(all_tables.items())[:int(length/4)]:
    #     all_results.append(func_(item))
    with Pool(n_threads) as p:
        func_ = partial(running_function,ttype=type)
        all_results = list(tqdm(p.imap(func_, list(all_tables.items())[:int(length/4)], chunksize=16), total=int(length/4),
                                desc="extract evidence chain", ))

    all_no_found = set()
    if type=='fake_pretrain_data':
        save_path = f'{basic_dir}/data/preprocessed_data/evidence_chain/pre-training/fake_question_pretraining'
        train_instances_cnt, dev_instances_cnt = 0,0
        # print(len(all_results))
        outputs_results_train, outputs_results_dev = data_split([result for results in all_results for result in results[0]],0.99,shuffle=True)
        with open(save_path + "_dev.jsonl", 'w', encoding='utf8') as outf:
            for item in outputs_results_dev:
                outf.write(json.dumps(item)+'\n')
                dev_instances_cnt+=1
        with open(save_path + "_train.jsonl", 'w', encoding='utf8') as outf:
            for item in outputs_results_train:
                outf.write(json.dumps(item)+'\n')
                train_instances_cnt+=1
        print('number of train instances {}, dev_instances {}'.format(train_instances_cnt, dev_instances_cnt))
    elif type=='bart_inference_data':
        save_path = f'{basic_dir}/data/preprocessed_data/evidence_chain/pre-training/inference_bart_evidence_chain_0_addneg.jsonl'

        instances_cnt = 0
        with open(save_path,'w',encoding='utf8') as outf:
            for results in all_results:
                all_no_found = all_no_found.union(results[1])
                for result in results[0]:
                    outf.write(json.dumps(result)+'\n')
                    instances_cnt+=1
        print('all no found passage: {}'.format(len(all_no_found)))
        print('number of instances {}'.format(instances_cnt))
    elif type=='pretrain_negatives':
        save_path = f'{basic_dir}/data/preprocessed_data/evidence_chain/pre-training/tbrep2negs_new.json'
        tbrep2negs = {}
        for results in all_results:
            tbrep2negs.update(results[2])
        outf = open(save_path,'w',encoding='utf8')
        print('length of tbrep 2 negs {}'.format(len(tbrep2negs)))
        print('Saving to {}'.format(save_path))
        json.dump(tbrep2negs,outf)

