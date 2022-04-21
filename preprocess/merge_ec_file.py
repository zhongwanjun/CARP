import json
import os
from tqdm import tqdm
import copy
from scipy.special import softmax
project_dir = './ODQA'
basic_dir = f'{project_dir}/data/evidence_chain_data/retrieval-based/scored_chains/'
#weighted
# gt_ec_file = f'{project_dir}/data/preprocessed_data/evidence_chain/ground-truth-based/scored_chains/train_gttb_evidence_chain_weighted_scores.json'
gt_ec_file = f'{project_dir}/data/evidence_chain_data/ground-truth-based/scored_chains/dev_roberta_large_scored_ec_addtab_weighted_multineg.json'
#original

file_names = ['dev_evidence_chain_weighted_scores.json']
all_data = []
gt_ec_data = json.load(open(gt_ec_file,'r',encoding='utf8'))
error_count = 0
all_cnt = 0

for fid,file in enumerate(file_names):
    start = 0 #dev
    # start = int(len(gt_ec_data) / 4) * fid
    end = len(gt_ec_data)#dev
    # end = int(len(gt_ec_data) / 4) * (fid + 1) if fid != (3) else len(gt_ec_data)
    data = json.load(open(os.path.join(basic_dir,file),'r',encoding='utf8'))
    output_data = []
    for idx, item in tqdm(enumerate(data)):
        assert (item['question_id'] == gt_ec_data[start + idx]['question_id'])
        output_data.append(copy.deepcopy(item))
        output_data[-1]['positive_table_blocks'] = copy.deepcopy(gt_ec_data[start + idx]['positive_table_blocks'])
        tmp_retrived_blocks = copy.deepcopy(item['retrieved_tbs'][:15])
        for tbid,block in enumerate(tmp_retrived_blocks):
            # assert(block['table_id']==gt_ec_data[start+idx]['retrieved_tbs'][tbid]['table_id'] and block['row_id']==gt_ec_data[start+idx]['retrieved_tbs'][tbid]['row_id'])
            all_cnt+=1
            if block['candidate_evidence_chains']:
                ranked_ec = sorted(block['candidate_evidence_chains'], key=lambda k: softmax(k['score'])[1], reverse=True)
                if len(ranked_ec)>=3:
                    selected = copy.deepcopy(ranked_ec[:3])
                else:
                    selected = copy.deepcopy(ranked_ec)
                del tmp_retrived_blocks[tbid]['candidate_evidence_chains']
                tmp_retrived_blocks[tbid]['candidate_evidence_chains'] = selected
            else:
                error_count+=1
        output_data[-1]['retrieved_tbs'] = tmp_retrived_blocks
        assert(len(output_data[-1]['retrieved_tbs'])==15)

    all_data.extend(output_data)
print('error rate: {}'.format(error_count/all_cnt))
with open(os.path.join(basic_dir,f'{project_dir}/data/qa_with_evidence_chain/dev_ranked_evidence_chain_for_qa_weighted.json'),'w',encoding='utf8') as outf:
    json.dump(all_data,outf)


