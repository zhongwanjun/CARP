import json
from tqdm import tqdm
from scipy.special import softmax
import sys

def convert_chain_to_string(chain):
    chain_rep = []
    for node in chain:
        # print(node)
        if node['origin']['where'] != 'question':
            if node['origin']['where'] == 'table':
                prefix = ' '
            elif node['origin']['where'] == 'passage':
                prefix = ' {} : '.format(
                    node['origin']['index'].replace('/wiki/', '').replace('_', ' ').split('/')[0])
            chain_rep.append(prefix + node['content'])
    # print(' [TO] '.join(chain_rep))
    # input()
    return ' ; '.join(chain_rep)
def find_union_chain(ranked_ec,topt=2):
    nodes = set()
    all_nodes = []
    for ec in ranked_ec[:topt]:
        # print(ec['path'])
        add = [node for node in ec['path'][1:] if node['content'] not in nodes]
        all_nodes.extend(add)
        nodes.update(set([node['content'] for node in ec['path'][1:]]))
    return all_nodes
from fuzzywuzzy import fuzz
def calculate_score(data):
    topk = {1:0,2:0,3:0,5:0,10:0,20:0}
    split = []
    all_selected_chain = []
    hit,all, length,union_hit,union_length=0,0,0,0,0
    for idx,d in tqdm(enumerate(data), desc='Calculating score'):
        positive_table_block = d['positive_table_blocks']

        for bid,block in enumerate(positive_table_block):
            best_score = -999
            selected_chain = ''
            # for chain in block['candidate_evidence_chains']:
            #     # positive_chain = convert_chain_to_string(chain['path'])
            #     score = softmax(chain['score'])#fuzz.partial_ratio(d['question']+' '+orig_answer,positive_chain)
            #     if score[1]>best_score:
            #         best_score = score[1]
            #         selected_chain = chain
            # for cid in range(len(block['candidate_evidence_chains'])):
            #     block['candidate_evidence_chains'][cid]['score'] = (softmax(block['candidate_evidence_chains'][cid]['score']) + softmax(data2[idx]['positive_table_blocks'][bid]['candidate_evidence_chains'][cid]['score']))/2
            ranked_ec = sorted(block['candidate_evidence_chains'],key=lambda k: k['score'][1],reverse=True)
            unied_ec = find_union_chain(ranked_ec,topt=2)
            # ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: k['score'][0], reverse=True)
            # print(ranked_ec)
            # input()
            for i in range(min(len(ranked_ec),20)):
                # print(ranked_ec[i]['path'][-1])
                if ranked_ec[i]['path']:
                    if any([node['is_end_node'] for node in ranked_ec[i]['path']]):
                        for j in topk.keys():
                            if i<j:
                                topk[j]+=1
                        break
            if unied_ec:
                union_length+=len(unied_ec)
                if any([node['is_end_node'] for node in unied_ec]):
                    union_hit+=1
            length += len(ranked_ec[0]['path']) if ranked_ec else 0
            # reranked_ec = []
            # for i in range(min(len(ranked_ec), 3)):
            #     # print(ranked_ec[i]['path'][-1])
            #     if ranked_ec[i]['path']:
            #         chain_rep = convert_chain_to_string(ranked_ec[i]['path'])
            #         question_rep = d['question']
            #         score = fuzz.token_set_ratio(question_rep.lower(),chain_rep.lower())
            #         ranked_ec[i]['fuzz_score'] = score
            #         reranked_ec.append(ranked_ec[i])
            #     else:
            #         ranked_ec[i]['fuzz_score'] = 0
            #         reranked_ec.append(ranked_ec[i])
            # reranked_ec = sorted(reranked_ec,key=lambda k:k['fuzz_score'],reverse=True)
            # if reranked_ec:#[0]['path']:
            #     if reranked_ec[0]['path'][-1]['is_end_node']:
            #         hit+=1
            all+=1
    print('Top 1 {}/{}={}'.format(topk[1],all,topk[1]/all))
    print('Top 3 {}/{}={}'.format(topk[2], all, topk[2] / all))
    print('Top 3 {}/{}={}'.format(topk[3], all, topk[3] / all))
    print('Top 5 {}/{}={}'.format(topk[5], all, topk[5] / all))
    print('Top 10 {}/{}={}'.format(topk[10], all, topk[10] / all))
    print('Top 3 Union {}/{}={}'.format(union_hit, all, union_hit/ all))
    print('Average length {}/{}={}'.format(length, all, length/ all))
    print('Average Union length {}/{}={}'.format(union_length, all, union_length / all))
basic_dir = '/home/t-wzhong/v-wanzho/ODQA/data'


# data_path = sys.argv[1]
# data_path='/home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/evidence_chain/ground-truth-based/dev_preprocessed_normalized_gtmodify_candidate_evichain_nx_scores_addnoun.json'
# data_path_2='/home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/evidence_chain/ground-truth-based/dev_preprocessed_normalized_gtmodify_candidate_evichain_nx_scores_addnoun.json'
# data_path = f'{basic_dir}/preprocessed_data/evidence_chain/dev_preprocessed_normalized_gtmodify_candidate_evichain_nx_scores_addnoun_roberta_tb.json' 
# data_path_2 = f'{basic_dir}/preprocessed_data/evidence_chain/dev_preprocessed_normalized_gtmodify_candidate_evichain_nx_scores_addnoun_roberta_tb.json'
# /home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/evidence_chain/dev_preprocessed_normalized_gtmodify_candidate_evichain_nx_scores_addnoun_roberta_tb.json
# file = sys.argv[1]
data_path = '/home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/evidence_chain/ground-truth-based/candidate_chain/../scored_chain/dev_roberta_base_scored_ec.json'
print(f"Loading data from {data_path}")
with open(data_path, 'r') as f:
    data = json.load(f)#[:100]
# with open(data_path_2, 'r') as f:
#     data_2 = json.load(f)#[:100]
# self.data = self.data[:300]
calculate_score(data)
# print(f"Total sample count {len(self.data)}")
