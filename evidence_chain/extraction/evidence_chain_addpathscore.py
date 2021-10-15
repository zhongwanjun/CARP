import nltk
import sys

sys.path.append('../../')
sys.path.append('../')
from rule_pattern import RulePattern
from fuzzywuzzy import fuzz
import json
from nltk.corpus import stopwords
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from networkx.algorithms.shortest_paths.generic import all_shortest_paths
from utils.common import find_similar_sentence_tfidf
stop_words = stopwords.words('english') + ['the','a','an']
def check_useful(kw):
    if kw.strip().isdigit():
        return False
    if kw in stop_words:
        return False
    return True
def cell_match(node1,node2):
    # cleaned_pid = passage_id.replace('/wiki/','').replace('_',' ').split('/')[0]
    # return fuzz.ratio(cleaned_pid,cell) > 70
    # node1 is table and node2 is passage
    n1_rc_id = ','.join(node1.orig['index'].split('/')[-2:])
    n2_rc_id = node2.orig['link_table']
    # print(n1_rc_id, n2_rc_id)
    return n1_rc_id == n2_rc_id


class Node():
    def __init__(self,node_id,content, kws, orig, end=False, neighbour=[]):
        self.node_id = node_id
        self.content = content
        self.orig = orig
        self.end= end
        self.kws = kws
        self.neighbour = neighbour
    def __repr__(self):
        rep = {'node_id':self.node_id,'content':self.content,'is_end_node':self.end,'keywords':self.kws,
               'neighbours':self.neighbour, 'origin':self.orig}
        return '{}'.format(json.dumps(rep,indent=4))
import networkx as nx

class EvidenceGraphNetworkx():
    def __init__(self,root=None,nodes=[],edges=[]):
        self.graph = nx.Graph()
        self.root = root
        self.graph.add_node(root)
        self.common_keywords = []
        self.cv = TfidfVectorizer(tokenizer=lambda s: s.split())
        for node in nodes:
            self.graph.add_node(node)
        for edge in edges:
            self.graph.add_edge(edge)

    def link_edges_weight(self):
        # print(self.edges)
        # print('using link edges weight')
        all_nodes = list(self.graph.nodes)
        # all_edges = []
        for idx1,node1 in enumerate(all_nodes):
            # print('length of nodes',len(self.nodes))
            for idx2,node2 in enumerate(all_nodes):
                if idx1!=idx2 and ((node1,node2) not in list(self.graph.edges)):
                    if (node1.orig['where']=='question'):
                        related, score = self.sentence_keywords_matching(node1.kws, node2.kws)
                        if related:
                            self.graph.add_edge(node1, node2,weight=score*100)
                            self.graph.add_edge(node2, node1, weight=score * 100)

                    elif (node1.orig['where'] == 'passage' and node2.orig['where'] == 'passage' and node1.orig['index'].split('/')[-2] == node2.orig['index'].split('/')[-2]) \
                            or (node1.orig['where'] == 'table' and node2.orig['where'] == 'table') \
                            or (node1.orig['where'] == 'table' and node2.orig['where'] == 'passage' and cell_match(node1, node2)):
                        # related_score = self.keyword_matching(node1.kws,node2.kws)
                        self.graph.add_edge(node2,node1,weight=10)
                        self.graph.add_edge(node1,node2,weight=10)
                        # print(idx2,node2.node_id)
                        # self.nodes[idx1].neighbour.append(node2.node_id)
                        # self.nodes[idx2].neighbour.append(node1.node_id)
                        # self.graph.add_edge(node1,node2)
                        # self.graph.add_edge(node2,node1)

    def link_edges(self):
        # print(self.edges)
        all_nodes = list(self.graph.nodes)
        for idx1,node1 in enumerate(all_nodes):
            # print('length of nodes',len(self.nodes))
            for idx2,node2 in enumerate(all_nodes):
                if idx1!=idx2 and ((node1,node2) not in list(self.graph.edges)):
                    if (    node1.orig['where']=='question' and self.sentence_keywords_matching(node1.kws,node2.kws)[0] \
                            or self.keyword_matching(node1.kws,node2.kws)) \
                            or (node1.orig['where'] == 'passage' and node2.orig['where'] == 'passage' and node1.orig['index'].split('/')[-2] == node2.orig['index'].split('/')[-2])\
                            or (node1.orig['where'] == 'table' and node2.orig['where'] == 'table')\
                            or (node1.orig['where'] == 'table' and node2.orig['where'] == 'passage' and cell_match(node1, node2)):
                        # print(idx2,node2.node_id)
                        # self.nodes[idx1].neighbour.append(node2.node_id)
                        # self.nodes[idx2].neighbour.append(node1.node_id)
                        self.graph.add_edge(node1,node2)
                        self.graph.add_edge(node2,node1)

    def build_graph(self, nodes):
        all_keywords = []
        for node in nodes:
            self.graph.add_node(node)
            all_keywords.extend(node.kws)
        c = Counter(all_keywords)
        self.common_keywords = [k.lower() for k in c.keys()][:5]
        # print(self.common_keywords)
        self.link_edges_weight()
        return

    def keyword_matching(self,kws1, kws2):
        low_kws1 = set([kw.lower() for kw in kws1 if kw.lower() not in stop_words+self.common_keywords])
        low_kws2 = set([kw.lower() for kw in kws2 if kw.lower() not in stop_words+self.common_keywords])
        inter = len(low_kws1.intersection(low_kws2))
        if inter > 0:
            return True, 1 - inter / min(len(low_kws1),len(low_kws2))
        # for kw1 in low_kws1:
        #     for kw2 in low_kws2:
        #         if check_useful(kw1) and check_useful(kw2) and fuzz.ratio(kw1,kw2) > 65:
        #             # print('match: ',kw1,'|',kw2)
        #             return True
        return False,1

    def sentence_keywords_matching(self,kws1, kws2):
        low_kws1 = set([kw.lower() for kw in kws1 if kw not in stop_words+self.common_keywords])
        # clean_low_kws1 = self.clean_keywords(low_kws1)
        low_kws2 = set([kw.lower() for kw in kws2 if kw not in stop_words+self.common_keywords] )
        hits = 0
        inter = low_kws1.intersection(low_kws2)
        if len(inter) > 0:
            return True, 1 - len(inter) / len(low_kws1)
        for kw1 in low_kws1:
            for kw2 in low_kws2:
                if check_useful(kw1) and check_useful(kw2) and (kw1 in kw2 or (fuzz.ratio(kw1, kw2) > 65)):
                    # print('match: ',kw1,'|',kw2)
                    hits+=1
                    break
        if hits>0:
            return True, 1 - hits/len(low_kws1)
        return False,1

    def convert_chain_to_string(self,chain):
        chain_rep = []
        for node in chain:
            # print(node)
            if node.orig['where']!='question':
                if node.orig['where'] == 'table':
                    prefix = ' '
                elif node.orig['where'] == 'passage':
                    prefix = ' {} : '.format(
                        node.orig['index'].replace('/wiki/', '').replace('_', ' ').split('/')[0])
                chain_rep.append(prefix + node.content)
        # print(' [TO] '.join(chain_rep))
        # input()
        return ' ; '.join(chain_rep)

    def find_shortest_paths(self,question,answer):
        output_paths = []
        for nid, node in enumerate(self.graph.nodes):
            if node.end:
                paths = [path for path in nx.all_shortest_paths(self.graph, source=self.root, target=node,weight='weight')]
                # path_reps = [self.convert_chain_to_string(chain) for chain in paths]
                # best_score, best_path = 0, []
                # for i, rep in enumerate(path_reps):
                #     score = fuzz.partial_ratio('{} {}'.format(question.lower(),answer.lower()), rep.lower())
                #     if score > best_score:
                #         best_score = score
                #         best_path = paths[i]
                # # print(paths)
                # output_paths.append(best_path)
                output_paths.extend(paths)
        # for path in output_paths:
        #     print(path)
        return output_paths
    def find_all_paths(self,question,answer,cutoff=4):
        output_paths = []
        for nid, node in enumerate(self.graph.nodes):
            if node.end:
                paths = [path for path in nx.all_simple_paths(self.graph, source=self.root, target=node,cutoff=cutoff)]
                hop_scores, match_scores = [], []
                paths = random.choices(paths,k=15)
                for path in paths:
                    prev_orig = path[0].orig['where']
                    hop_score, match_score  = 0,0
                    for nid,node in enumerate(path[1:]):
                        if nid!=0 and (node.orig['where']!=prev_orig or (node.orig['where']=='table' and prev_orig=='table')):
                            hop_score+=1
                        prev_orig = node.orig['where']
                    hop_scores.append(hop_score/len(path))
                    path_rep = self.convert_chain_to_string(path)
                    match_score = fuzz.token_sort_ratio('{} {}'.format(question.lower(), answer.lower()), path_rep.lower())
                    match_scores.append(match_score)
                normalized_hop_scores = np.array([s/(sum(hop_scores)+1e-30) for s in hop_scores])
                normalized_match_scores = np.array([s /(sum(match_scores)+1e-30) for s in match_scores])
                path_scores = (normalized_match_scores)/2
                score_paths = [(path,score) for path, score in zip(paths,path_scores.tolist())]
                score_paths = sorted(score_paths,key=lambda k: k[1],reverse=True)
                if len(score_paths)>=4:
                    output_paths.extend([item[0] for item in score_paths[:4]])
                else:
                    output_paths.extend([item[0] for item in score_paths])
        # for path in output_paths:
        #     print(path)
        return output_paths

    def find_path_to_all_nodes(self,question):

        candidate_paths = []
        # node2path = nx.shortest_path(self.graph, source=self.root)
        # node2path = nx.single_source_shortest_path(self.graph, source=self.root,cutoff=4)
        for nid, node in enumerate(self.graph.nodes):
            node_paths = [path for path in nx.all_shortest_paths(self.graph, source=self.root, target=node, weight='weight')]
            path_reps = [self.convert_chain_to_string(chain) for chain in node_paths]
            best_score, best_path = 0,[]
            for i,rep in enumerate(path_reps):
                score = fuzz.token_sort_ratio(question.lower(),rep.lower())
                if score > best_score:
                    best_score = score
                    best_path = node_paths[i]
            candidate_paths.append(best_path)
            # candidate_paths.extend(node_paths)
        return candidate_paths

    def find_negative_paths(self,paths,question,answer):
        negative_paths = []
        for pid,path in enumerate(paths):
            path = list(path)
            end_node = random.choice(list(self.graph.nodes))
            while end_node.end:
                end_node = random.choice(list(self.graph.nodes))
            neg_paths = [path for path in nx.all_shortest_paths(self.graph, source=self.root, target=end_node,weight='weight')]
            path_reps = [self.convert_chain_to_string(chain) for chain in neg_paths]

            best_score, best_path = 0, []
            for i, rep in enumerate(path_reps):
                score = fuzz.partial_ratio('{} {}'.format(question.lower(), answer.lower()), rep.lower())
                if score > best_score:
                    best_score = score
                    best_path = neg_paths[i]

            negative_paths.append(best_path)
        # print(paths)
        # print(negative_paths)
        # input()
        return negative_paths
    def find_multiple_neg_paths(self,paths,question,answer):
        negative_paths = []
        all_negative_paths = []
        neg_end_nodes = [node for node in list(self.graph.nodes) if not node.end]
        for neg_node in neg_end_nodes:
            all_negative_paths.extend([path for path in nx.all_shortest_paths(self.graph, source=self.root, target=neg_node, weight='weight')
                                       if not any([n.end for n in path])])
        all_negative_path_reps = [self.convert_chain_to_string(chain) for chain in all_negative_paths]
        for pid, path in enumerate(paths):
            pos_rep = self.convert_chain_to_string(path)
            neg_paths = find_similar_sentence_tfidf(pos_rep,all_negative_path_reps,all_negative_paths,self.cv,topk=5)
            negative_paths.append(neg_paths)
        return negative_paths
    def represent_paths(self,paths,mode='multiple'):
        all_path_rep = []
        for p in paths:
            if mode=='multiple':
                ppath_rep = []
                for subp in p:
                    path_rep = []
                    for node in subp:
                        path_rep.append({'content': node.content, 'origin': node.orig, 'node_id': node.node_id,
                                         'keywords': '{}'.format(node.kws), 'is_end_node': node.end})
                    ppath_rep.append(path_rep)
                all_path_rep.append(ppath_rep)
            else:
                path_rep = []
                for node in p:
                    path_rep.append({'content':node.content,'origin':node.orig,'node_id':node.node_id,
                                     'keywords':'{}'.format(node.kws),'is_end_node':node.end})
                all_path_rep.append(path_rep)
        return all_path_rep



def clean_keywords(keywords):
    cleaned = []
    for keyword in keywords:
        words = [wd for wd in nltk.word_tokenize(keyword) if wd.lower() not in stop_words]
        if words:
            cleaned.append(' '.join(words))
    return cleaned

def merge_keywords(kws):
    all_kws = list(set(kws['subject']+kws['entity']+kws['numbers']+kws['noun']))
    cleaned = clean_keywords(all_kws)
    return cleaned#list(set(kws['subject']+kws['entity']+kws['numbers']+kws['noun']))

def extract_question_kws(question,RP):
    # print(question)
    q_kws = RP.found_key_words([question])[0]
    # print(q_kws) 
    return q_kws

def extract_key_words_retrived_tb_multiple(table_blocks,RP):
    all_table_values = []
    all_sens = []
    all_p_sens = {}
    tbid2tab = {}
    tbid2psens = {}
    start_t,start_s = 0,0
    for tb_id,table_block in enumerate(table_blocks):
        table_row, passages = table_block['table'], table_block['passages']
        if passages:
            assert 'link_table' in passages[0].keys()
        table_values = [item if item else 'none' for item in table_row[1][0]]
        p_sens = [nltk.sent_tokenize(doc['passage']) if doc else [] for doc in passages]
        sens = [s for s_list in p_sens for s in s_list]
        all_p_sens[tb_id] = (p_sens)
        all_sens.extend(sens)
        all_table_values.extend(table_values)
        tbid2tab[tb_id] = (start_t,start_t+len(table_values))
        start_t+=len(table_values)
        tbid2psens[tb_id] = []
        for pidx, p_sens in enumerate(p_sens):
            if p_sens:
                tbid2psens[tb_id].append((start_s,start_s+len(p_sens)))
            else:
                tbid2psens[tb_id].append((-1,-1))
            start_s+=len(p_sens)
    all_keywords = RP.found_key_words(all_table_values+all_sens )
    # all_sen_keywords = RP.found_key_words(all_sens)
    for tb_id, table_block in enumerate(table_blocks):
        table_v_kws = all_keywords[tbid2tab[tb_id][0]:tbid2tab[tb_id][1]]
        psgs_kws = []
        p_spans = tbid2psens[tb_id]
        new_start = len(all_table_values)
        for span in p_spans:
            if span[0]!=-1:
                psgs_kws.append(all_keywords[new_start+span[0]:new_start+span[1]])
            else:
                psgs_kws.append([])
        #check
        table_row, passages = table_block['table'], table_block['passages']
        table_values = [item if item else 'none' for item in table_row[1][0]]
        assert (len(table_v_kws) == len(table_values)), 'table kws length {}/ table value length {}'.format(len(table_v_kws), len(table_values))
        assert (all([table_values[vid] == kws['content'] for vid, kws in enumerate(table_v_kws)])), 'values: {}, kws: {}'.format(table_values,
                                                                             [item['content'] for item in table_v_kws])
        # print(all_p_sens[tb_id],psgs_kws)
        assert (all([sen == psgs_kws[pid][sid]['content'] for pid in range(len(all_p_sens[tb_id])) for sid,sen in enumerate(all_p_sens[tb_id][pid])])), 'sentence: {}, context: {}'.format(all_p_sens[tb_id],psgs_kws)
        #end of check
        table_blocks[tb_id]['passage_keywords'] = psgs_kws
        table_blocks[tb_id]['table_value_keywords'] = table_v_kws
    return table_blocks

def extract_key_words_multiple(table_blocks,RP):
    all_table_values = []
    all_sens = []
    all_p_sens = {}
    tbid2tab = {}
    tbid2psens = {}
    start_t,start_s = 0,0
    for tb_id,table_block in enumerate(table_blocks):
        table_row, passages = table_block['table_segment'],table_block['gt_passages']
        # if passages:
        #     assert 'link_table' in passages[0].keys()
        # table_values = [item if item else 'none' for item in table_row[1][0]]
        table_values = [item['0'] if item['0'] else 'empty' for item in table_row.values()]
        p_sens = [nltk.sent_tokenize(doc['passage']) if doc else [] for doc in passages]
        sens = [s for s_list in p_sens for s in s_list]
        all_p_sens[tb_id] = (p_sens)
        all_sens.extend(sens)
        all_table_values.extend(table_values)
        tbid2tab[tb_id] = (start_t,start_t+len(table_values))
        start_t+=len(table_values)
        tbid2psens[tb_id] = []
        for pidx, p_sens in enumerate(p_sens):
            if p_sens:
                tbid2psens[tb_id].append((start_s,start_s+len(p_sens)))
            else:
                tbid2psens[tb_id].append((-1,-1))
            start_s+=len(p_sens)
    all_keywords = RP.found_key_words(all_table_values+all_sens )
    # all_sen_keywords = RP.found_key_words(all_sens)
    for tb_id, table_block in enumerate(table_blocks):
        table_v_kws = all_keywords[tbid2tab[tb_id][0]:tbid2tab[tb_id][1]]
        psgs_kws = []
        p_spans = tbid2psens[tb_id]
        new_start = len(all_table_values)
        for span in p_spans:
            if span[0]!=-1:
                psgs_kws.append(all_keywords[new_start+span[0]:new_start+span[1]])
            else:
                psgs_kws.append([])
        #check
        table_row, passages = table_block['table_segment'], table_block['passages']
        table_values = [item['0'] if item['0'] else 'empty' for item in table_row.values()]
        assert (len(table_v_kws) == len(table_values)), 'table kws length {}/ table value length {}'.format(len(table_v_kws), len(table_values))
        assert (all([table_values[vid] == kws['content'] for vid, kws in enumerate(table_v_kws)])), 'values: {}, kws: {}'.format(table_values,
                                                                             [item['content'] for item in table_v_kws])
        # print(all_p_sens[tb_id],psgs_kws)
        assert (all([sen == psgs_kws[pid][sid]['content'] for pid in range(len(all_p_sens[tb_id])) for sid,sen in enumerate(all_p_sens[tb_id][pid])])), 'sentence: {}, context: {}'.format(all_p_sens[tb_id],psgs_kws)
        #end of check
        table_blocks[tb_id]['passage_keywords'] = psgs_kws
        table_blocks[tb_id]['table_value_keywords'] = table_v_kws
    return table_blocks

def extract_key_words_retrived_tb(table_row,passages,RP):
    all_passage_kws = []
    table_values = table_row[1][0]
    # print('table_values',table_values)
    # print(table_values)
    table_values = [item if item else 'none' for item in table_row[1][0]]
    # table_v_kws = RP.found_key_words(table_values)
    # print('table keywords', table_v_kws)
    all_p_sens = [nltk.sent_tokenize(doc['passage']) if doc else [] for doc in passages]
    all_sens = [s for s_list in all_p_sens for s in s_list]
    # print(all_sens)

    all_keywords = RP.found_key_words(table_values + all_sens)
    table_v_kws = all_keywords[0:len(table_values)]
    assert (len(table_v_kws) == len(table_values)), 'table kws length {}/ table value length {}'.format(
        len(table_v_kws), len(table_values))
    assert (all([table_values[vid] == kws['content'] for vid,kws in enumerate(table_v_kws)])),'values: {}, kws: {}'.format(table_values,[item['content'] for item in table_v_kws])
    assert (len(table_values+all_sens) == len(all_keywords)), 'passage kws number {}/ sentence number {}'.format(
        len(all_keywords), len(all_sens+table_values))
    start = len(table_values)
    for pidx, p_sens in enumerate(all_p_sens):
        if p_sens:
            all_passage_kws.append(all_keywords[start:start + len(p_sens)])
            start += len(p_sens)
            for sid, s in enumerate(p_sens):
                assert (s == all_passage_kws[-1][sid]['content']), 'sentence: {}, context: {}'.format(s,all_passage_kws[-1][sid]['content'])
        else:
            all_passage_kws.append([])
    # for doc in passages:
    #     if doc:
    #         doc = doc['passage']
    #         sens = nltk.sent_tokenize(doc)
    #         # print('sentences',sens)
    #         all_keywords = RP.found_key_words(sens)
    #         # print('sentence keywords',all_keywords)
    #         all_passage_kws.append(all_keywords)
    #         assert (len(all_keywords) == len(sens)), 'passage kws number {}/ sentence number {}'.format(
    #             len(all_keywords), len(sens))
    #     else:
    #         all_passage_kws.append([])
    return all_passage_kws, table_v_kws
def extract_key_words( table_row, passages,RP):
    all_passage_kws = []
    table_values = [item['0'] if item['0'] else 'empty' for item in table_row.values()]
    # print('table_values',table_values)
    # table_v_kws = RP.found_key_words(table_values)
    # assert(len(table_v_kws)==len(table_values)),'table kws length {}/ table value length {}'.format(len(table_v_kws),len(table_values))
    # print('table keywords', table_v_kws)
    all_p_sens = [nltk.sent_tokenize(doc['passage']) for doc in passages]
    all_sens = [s for s_list in all_p_sens for s in s_list]
    # print(all_sens)
    all_keywords = RP.found_key_words(table_values + all_sens)
    table_v_kws = all_keywords[0:len(table_values)]
    assert (len(table_v_kws) == len(table_values)), 'table kws length {}/ table value length {}'.format(
        len(table_v_kws), len(table_values))
    assert (all(
        [table_values[vid] == kws['content'] for vid, kws in enumerate(table_v_kws)])), 'values: {}, kws: {}'.format(
        table_values, [item['content'] for item in table_v_kws])
    assert (len(table_values + all_sens) == len(all_keywords)), 'passage kws number {}/ sentence number {}'.format(
        len(all_keywords), len(all_sens + table_values))
    start = len(table_values)
    for pidx, p_sens in enumerate(all_p_sens):
        if p_sens:
            all_passage_kws.append(all_keywords[start:start + len(p_sens)])
            start += len(p_sens)
            for sid, s in enumerate(p_sens):
                assert (s == all_passage_kws[-1][sid]['content']), 'sentence: {}, context: {}'.format(s,
                                                                                                      all_passage_kws[
                                                                                                          -1][sid][
                                                                                                          'content'])
        else:
            all_passage_kws.append([])

    # all_keywords = RP.found_key_words(all_sens)
    # assert (len(all_sens) == len(all_keywords)), 'passage kws number {}/ sentence number {}'.format(
    #     len(all_keywords), len(all_sens))
    # start = 0
    # for pidx, p_sens in enumerate(all_p_sens):
    #     all_passage_kws.append(all_keywords[start:start+len(p_sens)])
    #     start+=len(p_sens)
    #     for sid,s in p_sens:
    #         assert(s==all_passage_kws[-1][sid]['content']),'sentence: {}, context: {}'.format(s,all_passage_kws[-1][sid]['content'])
    '''
    for doc in passages:
        sens = nltk.sent_tokenize(doc['passage'])
        # print('sentences',sens)
        all_keywords = RP.found_key_words(sens)
        # print('sentence keywords',all_keywords)
        all_passage_kws.append(all_keywords)
        assert (len(all_keywords) == len(sens)), 'passage kws number {}/ sentence number {}'.format(
            len(all_keywords), len(sens))
    '''
    return all_passage_kws,table_v_kws


def extract_evidence_chain(data,table_block):

    q_kws = data['question_keywords']#RP.found_key_words([question])[0]
    nodes = []
    node_id = 0

    root = Node(node_id=node_id, content=data['question'], kws=merge_keywords(q_kws), orig={'where': 'question', 'index': None},end=False,neighbour=[])
    Graph = EvidenceGraphNetworkx(root,[],[])
    node_id+=1
    passages = table_block['gt_passages']
    for doc_id,doc in enumerate(passages):
        # sens = nltk.sent_tokenize(doc['passage'])
        # try:
        sen_keywords = table_block['gt_passages_keywords'][doc_id]

        # if data['where'] in ['passage','both']:
        for sid, sen_kws in enumerate(sen_keywords):
            sen = sen_kws['content']
            start = sen.lower().find(data['answer-text'].lower())
            # print(sen_kws)
            if start != -1:
                nodes.append(Node(node_id=node_id,content=sen,kws=merge_keywords(sen_kws),
                                  orig={'where':'passage','link_table':doc['link_table'],'index':'{}/{}'.format(doc['passage_id'],sid)},
                                      end=True,neighbour=[]))
            else:
                nodes.append(Node(node_id=node_id,content=sen, kws=merge_keywords(sen_kws),
                                      orig={'where': 'passage', 'link_table':doc['link_table'],'index': '{}/{}'.format(doc['passage_id'], sid)},
                                      end=False,neighbour=[]))
            node_id+=1
    # try:

    table_v_kws = table_block['table_value_keywords']#RP.found_key_words([item['0'] for item in table_row.values()])
    table_row = table_block['table_segment']
    # except Exception as e:
    #     print(e) 
    #     table_v_kws = [{'noun': [], 'claim': v , 'subject': [], 'entity': []} for v in table_row.values()]
    table_title = table_block['table_id'].replace('_',' ')
    for cell_id,(k,v) in enumerate(table_row.items()):
        v=v['0']
        if (data['where'] in ['table','both']) and (v.lower().find(data['answer-text'].lower())!=-1):
            nodes.append(Node(node_id=node_id,content=f'{k} is {v}',kws=[k]+merge_keywords(table_v_kws[cell_id]),orig={'where':'table','index':'{}/{}/{}'.format(table_block['table_id'],table_block['row_id'],cell_id)},end=True,neighbour=[]))
        else:
            nodes.append(Node(node_id=node_id,content=f'{k} is {v}',kws=[k]+ merge_keywords(table_v_kws[cell_id]),orig={'where':'table','index':'{}/{}/{}'.format(table_block['table_id'],table_block['row_id'],cell_id)},end=False,neighbour=[]))
        node_id+=1


    Graph.build_graph(nodes)
    # non_link = 0
    if list(Graph.graph.neighbors(Graph.root)):
        # print('gt passages', table_block['gt_passages'])
        # print(table_block['gt_passages_keywords'])
        # print('passages', table_block['passages'])
        # for n in Graph.nodes:
        #     print(n)
        try:
            shortest_paths = Graph.find_shortest_paths(data['question'],data['answer-text'])
            negative_paths = Graph.find_multiple_neg_paths(shortest_paths,data['question'],data['answer-text'])
            all_paths = Graph.find_all_paths(data['question'],data['answer-text'],cutoff=4)
            all_paths_representation = Graph.represent_paths(all_paths,'single')
            positive_paths_representation = Graph.represent_paths(shortest_paths,'single')
            if all_paths_representation:
                positive_paths_representation.append(all_paths_representation[0])
            negative_paths_representation = Graph.represent_paths(negative_paths,'multiple')
        except Exception as e:
            print(e)
            print(list(Graph.graph.neighbors(Graph.root)))
            # print(Graph.graph.nodes)
            print('#########################################')
            print([(edge[0].node_id,edge[1].node_id) for edge in Graph.graph.edges])
            print('#########################################')
            positive_paths_representation = []
            negative_paths_representation = []
            all_paths_representation = []

        # print('the answer is',data['answer-text'])
        # for rep in paths_representation:
        #     print(rep)
        # print('--------------------------------------------------')
    else:
        # print('QA pair is: {} / {}'.format(data['question'], data['answer-text']))
        # print('Question Keywords: {}'.format(q_kws))
        # for n in Graph.graph.nodes:
        #     print(n)
        # print('----------------------------------')
        # print('#########################################')
        # print([(edge[0].node_id, edge[1].node_id) for edge in Graph.graph.edges])
        # print('#########################################')
        positive_paths_representation = []
        negative_paths_representation = []
        all_paths_representation = []
        # non_link+=1
    # print(non_link)

    return positive_paths_representation, negative_paths_representation, all_paths_representation

def extract_evidence_chains_for_ranking_retrived(data, table_block):
    q_kws = data['question_keywords']  # RP.found_key_words([question])[0]
    nodes = []
    node_id = 0

    root = Node(node_id=node_id, content=data['question'], kws=merge_keywords(q_kws),
                orig={'where': 'question', 'index': None}, end=False, neighbour=[])
    Graph = EvidenceGraphNetworkx(root, [], [])
    node_id += 1
    passages = table_block['passages']
    for doc_id, doc in enumerate(passages):
        # sens = nltk.sent_tokenize(doc['passage'])
        # try:
        sen_keywords = table_block['passages']['keywords'][doc_id]

        # if data['where'] in ['passage','both']:
        for sid, sen_kws in enumerate(sen_keywords):
            sen = sen_kws['content']
            start = sen.lower().find(data['answer-text'].lower())
            # print(sen_kws)
            if start != -1:
                nodes.append(Node(node_id=node_id, content=sen, kws=merge_keywords(sen_kws),
                                  orig={'where': 'passage', 'link_table': doc['link_table'],
                                        'index': '{}/{}'.format(doc['passage_id'], sid)},
                                  end=True, neighbour=[]))
            else:
                nodes.append(Node(node_id=node_id, content=sen, kws=merge_keywords(sen_kws),
                                  orig={'where': 'passage', 'link_table': doc['link_table'],
                                        'index': '{}/{}'.format(doc['passage_id'], sid)},
                                  end=False, neighbour=[]))
            node_id += 1
    # try:

    table_v_kws = table_block['table_value_keywords']  # RP.found_key_words([item['0'] for item in table_row.values()])
    table_row = table_block['table_segment']
    # except Exception as e:
    #     print(e)
    #     table_v_kws = [{'noun': [], 'claim': v , 'subject': [], 'entity': []} for v in table_row.values()]
    table_title = table_block['table_id'].replace('_', ' ')
    for cell_id, (k, v) in enumerate(table_row.items()):
        v = v['0']
        if (data['where'] in ['table', 'both']) and (v.lower().find(data['answer-text'].lower()) != -1):
            nodes.append(Node(node_id=node_id, content=f'{k} is {v}', kws=[k] + merge_keywords(table_v_kws[cell_id]),
                              orig={'where': 'table',
                                    'index': '{}/{}/{}'.format(table_block['table_id'], table_block['row_id'],
                                                               cell_id)}, end=True, neighbour=[]))
        else:
            nodes.append(Node(node_id=node_id, content=f'{k} is {v}', kws=[k] + merge_keywords(table_v_kws[cell_id]),
                              orig={'where': 'table',
                                    'index': '{}/{}/{}'.format(table_block['table_id'], table_block['row_id'],
                                                               cell_id)}, end=False, neighbour=[]))
        node_id += 1

    Graph.build_graph(nodes)
    # non_link = 0
    if list(Graph.graph.neighbors(Graph.root)):
        # print('gt passages', table_block['gt_passages'])
        # print(table_block['gt_passages_keywords'])
        # print('passages', table_block['passages'])
        # for n in Graph.nodes:
        #     print(n)
        try:
            candidate_paths = Graph.find_path_to_all_nodes(data['question'])
            candidate_paths_representation = Graph.represent_paths(candidate_paths)
        except Exception as e:
            print(e)
            print(list(Graph.graph.neighbors(Graph.root)))
            print(Graph.graph.nodes)
            print('#########################################')
            print([(edge[0].node_id, edge[1].node_id) for edge in Graph.graph.edges])
            print('#########################################')
            candidate_paths_representation = []

        # print('the answer is',data['answer-text'])
        # for rep in paths_representation:
        #     print(rep)
        # print('--------------------------------------------------')
    else:
        # for n in Graph.nodes:
        #     print(n)
        # print('----------------------------------')
        candidate_paths_representation = []
        # non_link+=1
    # print(non_link)

    return candidate_paths_representation
def extract_evidence_chains_for_ranking(data,table_block):

    q_kws = data['question_keywords']#RP.found_key_words([question])[0]
    nodes = []
    node_id = 0

    root = Node(node_id=node_id, content=data['question'], kws=merge_keywords(q_kws), orig={'where': 'question', 'index': None},end=False,neighbour=[])
    Graph = EvidenceGraphNetworkx(root,[],[])
    node_id+=1
    passages = table_block['gt_passages']
    for doc_id,doc in enumerate(passages):
        # sens = nltk.sent_tokenize(doc['passage'])
        # try:
        sen_keywords = table_block['gt_passages_keywords'][doc_id]

        # if data['where'] in ['passage','both']:
        for sid, sen_kws in enumerate(sen_keywords):
            sen = sen_kws['content']
            start = sen.lower().find(data['answer-text'].lower())
            # print(sen_kws)
            if start != -1:
                nodes.append(Node(node_id=node_id,content=sen,kws=merge_keywords(sen_kws),
                                  orig={'where':'passage','link_table':doc['link_table'],'index':'{}/{}'.format(doc['passage_id'],sid)},
                                      end=True,neighbour=[]))
            else:
                nodes.append(Node(node_id=node_id,content=sen, kws=merge_keywords(sen_kws),
                                      orig={'where': 'passage', 'link_table':doc['link_table'],'index': '{}/{}'.format(doc['passage_id'], sid)},
                                      end=False,neighbour=[]))
            node_id+=1
    # try:

    table_v_kws = table_block['table_value_keywords']#RP.found_key_words([item['0'] for item in table_row.values()])
    table_row = table_block['table_segment']
    # except Exception as e:
    #     print(e)
    #     table_v_kws = [{'noun': [], 'claim': v , 'subject': [], 'entity': []} for v in table_row.values()]
    table_title = table_block['table_id'].replace('_',' ')
    for cell_id,(k,v) in enumerate(table_row.items()):
        v=v['0']
        if (data['where'] in ['table','both']) and (v.lower().find(data['answer-text'].lower())!=-1):
            nodes.append(Node(node_id=node_id,content=f'{k} is {v}',kws=[k]+merge_keywords(table_v_kws[cell_id]),orig={'where':'table','index':'{}/{}/{}'.format(table_block['table_id'],table_block['row_id'],cell_id)},end=True,neighbour=[]))
        else:
            nodes.append(Node(node_id=node_id,content=f'{k} is {v}',kws=[k]+ merge_keywords(table_v_kws[cell_id]),orig={'where':'table','index':'{}/{}/{}'.format(table_block['table_id'],table_block['row_id'],cell_id)},end=False,neighbour=[]))
        node_id+=1


    Graph.build_graph(nodes)
    # non_link = 0
    if list(Graph.graph.neighbors(Graph.root)):
        # print('gt passages', table_block['gt_passages'])
        # print(table_block['gt_passages_keywords'])
        # print('passages', table_block['passages'])
        # for n in Graph.nodes:
        #     print(n)
        try:
            candidate_paths = Graph.find_path_to_all_nodes(data['question'])
            candidate_paths_representation = Graph.represent_paths(candidate_paths,'single')
        except Exception as e:
            print(e)
            # print(list(Graph.graph.neighbors(Graph.root)))
            # print(Graph.graph.nodes)
            # print('#########################################')
            # print([(edge[0].node_id,edge[1].node_id) for edge in Graph.graph.edges])
            # print('#########################################')
            candidate_paths_representation = []

        # print('the answer is',data['answer-text'])
        # for rep in paths_representation:
        #     print(rep)
        # print('--------------------------------------------------')
    else:
        # for n in Graph.nodes:
        #     print(n)
        # print('----------------------------------')
        candidate_paths_representation = []
        # non_link+=1
    # print(non_link)

    return candidate_paths_representation

def extract_evidence_chains_for_ranking_retrived(data,table_block):

    q_kws = data['question_keywords']#RP.found_key_words([question])[0]
    nodes = []
    node_id = 0

    root = Node(node_id=node_id, content=data['question'], kws=merge_keywords(q_kws), orig={'where': 'question', 'index': None},end=False,neighbour=[])
    Graph = EvidenceGraphNetworkx(root,[],[])
    node_id+=1
    passages = table_block['passages']
    for doc_id,doc in enumerate(passages):
        # sens = nltk.sent_tokenize(doc['passage'])
        # try:
        sen_keywords = table_block['passage_keywords'][doc_id]

        # if data['where'] in ['passage','both']:
        for sid, sen_kws in enumerate(sen_keywords):
            sen = sen_kws['content']
            if 'answer-text' in data.keys():
                start = sen.lower().find(data['answer-text'].lower())
                # print(sen_kws)
                if start != -1:
                    nodes.append(Node(node_id=node_id,content=sen,kws=merge_keywords(sen_kws),
                                      orig={'where':'passage','link_table':doc['link_table'],'index':'{}/{}'.format(doc['index'],sid)},
                                          end=True,neighbour=[]))
                else:
                    nodes.append(Node(node_id=node_id,content=sen, kws=merge_keywords(sen_kws),
                                          orig={'where': 'passage', 'link_table':doc['link_table'],'index': '{}/{}'.format(doc['index'], sid)},
                                          end=False,neighbour=[]))
            else:
                nodes.append(Node(node_id=node_id, content=sen, kws=merge_keywords(sen_kws),
                                  orig={'where': 'passage', 'link_table': doc['link_table'],
                                        'index': '{}/{}'.format(doc['index'], sid)},
                                  end=False, neighbour=[]))
            node_id+=1
    # try:

    table_v_kws = table_block['table_value_keywords']#RP.found_key_words([item['0'] for item in table_row.values()])
    table_row = table_block['table'][1][0]
    table_header = table_block['table'][0]
    for cell_id,(v) in enumerate(table_row):
        if 'answer-text' in data.keys():
            start = v.lower().find(data['answer-text'].lower())
            if start != -1:
                nodes.append(Node(node_id=node_id,content=f'{table_header[cell_id]} is {v}',kws=[table_header[cell_id]]+ merge_keywords(table_v_kws[cell_id]),
                                  orig={'where':'table','index':'{}/{}/{}'.format(table_block['table_id'],table_block['row_id'],cell_id)},end=True,neighbour=[]))
            else:
                nodes.append(Node(node_id=node_id, content=f'{table_header[cell_id]} is {v}',
                                  kws=[table_header[cell_id]] + merge_keywords(table_v_kws[cell_id]),
                                  orig={'where': 'table',
                                        'index': '{}/{}/{}'.format(table_block['table_id'], table_block['row_id'],
                                                                   cell_id)}, end=False, neighbour=[]))
        else:
            nodes.append(Node(node_id=node_id, content=f'{table_header[cell_id]} is {v}',
                              kws=[table_header[cell_id]] + merge_keywords(table_v_kws[cell_id]),
                              orig={'where': 'table',
                                    'index': '{}/{}/{}'.format(table_block['table_id'], table_block['row_id'],
                                                               cell_id)}, end=False, neighbour=[]))
        node_id+=1


    Graph.build_graph(nodes)
    # non_link = 0
    if list(Graph.graph.neighbors(Graph.root)):
        # print('gt passages', table_block['gt_passages'])
        # print(table_block['gt_passages_keywords'])
        # print('passages', table_block['passages'])
        # for n in Graph.nodes:
        #     print(n)
        try:
            candidate_paths = Graph.find_path_to_all_nodes(data['question'])
            candidate_paths_representation = Graph.represent_paths(candidate_paths,'single')
        except Exception as e:
            print(e)
            # print(list(Graph.graph.neighbors(Graph.root)))
            # print(Graph.graph.nodes)
            # print('#########################################')
            # print([(edge[0].node_id,edge[1].node_id) for edge in Graph.graph.edges])
            # print('#########################################')
            candidate_paths_representation = []

        # print('the answer is',data['answer-text'])
        # for rep in paths_representation:
        #     print(rep)
        # print('--------------------------------------------------')
    else:
        # for n in Graph.nodes:
        #     print(n)
        # print('----------------------------------')
        candidate_paths_representation = []
        # non_link+=1
    # print(non_link)

    return candidate_paths_representation
def convert2num(string):
    string = string.replace(',', '')
    if string.endswith('%'):
        string = string.rstrip('%')
    try:
        string = float(string)
        return string
    except Exception:
        return None
from dateparser import parse
def find_superlative(table):

    mapping = {}
    headers = [_[0] for _ in table['header']]
    for j in range(len(table['header'])):
        mapping[headers[j]] = []
        activate_date_or_num = None
        if headers[j] not in ['#', 'Type', 'Name', 'Location', 'Position', 'Category', 'Nationality',
                              'School', 'Notes', 'Notability', 'Country']:
            for i, row in enumerate(table['data']):
                data = table['data'][i][j][0]
                if data in ['', '-']:
                    continue

                num = convert2num(data)
                if num and data.isdigit() and num > 1000 and num < 2020 and activate_date_or_num in ['date', None]:
                    date_format = parse(data)
                    mapping[headers[j]].append((date_format, 'date', [data, (i, j), None, None, 1.0]))
                    activate_date_or_num = 'date'
                elif num and activate_date_or_num in ['num', None]:
                    mapping[headers[j]].append((num, 'number', [data, (i, j), None, None, 1.0]))
                    activate_date_or_num = 'num'
                else:
                    try:
                        date_format = parse(data)
                        if date_format and activate_date_or_num in ['date', None]:
                            mapping[headers[j]].append((date_format, 'date', [data, (i, j), None, None, 1.0]))
                            activate_date_or_num = 'date'
                    except Exception:
                        continue

        if len(mapping[headers[j]]) < 0.3 * len(table['data']):
            mapping[headers[j]] = []

    nodes = []
    for k, v in mapping.items():
        if len(v) > 0:
            tmp = sorted(v, key = lambda x: x[0])
            if tmp[0][1] == 'number':
                tmp_node = tmp[0][-1]
                tmp_node[3] = 'minimum'
                nodes.append(tmp_node)
                tmp_node = tmp[-1][-1]
                tmp_node[3] = 'maximum'
                nodes.append(tmp_node)
            else:
                tmp_node = tmp[0][-1]
                tmp_node[3] = 'earliest'
                nodes.append(tmp_node)
                tmp_node = tmp[-1][-1]
                tmp_node[3] = 'latest'
                nodes.append(tmp_node)

    return nodes