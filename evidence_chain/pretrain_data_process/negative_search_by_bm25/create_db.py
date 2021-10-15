from elastic_search_wanjun import MyElastic
import os
import shutil
import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    args = parser.parse_args()
    ES=MyElastic()
    ES.delete_one()
    ES.create()
    # file_path = '/home/t-wzhong/v-wanzho/ODQA/data/data_wikitable/all_passages.json'
    if args.pretrain:
        file_path = '/home/t-wzhong/table-odqa/Data/evidence_chain/pre-training/evidence_output_pretrain_shortest.json'
        res = ES.bulk_insert_all_chains_pretrain(file_path)
    if args.finetune:
        basic_dir = '/home/t-wzhong/v-wanzho/ODQA/data/preprocessed_data/evidence_chain/ground-truth-based/ground-truth-evidence-chain/'
        file_paths = [os.path.join(basic_dir,file) for file in ['train_gt-ec-weighted.json','dev_gt-ec-weighted.json']]
        res = ES.bulk_insert_all_chains_finetune(file_paths)

    # res = ES.bulk_insert_all_doc(basic_path,file_path_list)
    # print(res)
    # print(len(res['hits']['hits']))

