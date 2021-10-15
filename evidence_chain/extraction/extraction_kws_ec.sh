#extract keywords for retrieved or ground-truth evidence block
python extract_evidence_chain.py --split train/dev --extract_keywords --kw_extract_type ground-truth/retrieved
#extract ground-truth evidence chain
python extract_evidence_chain.py --split train/dev --extract_evidence_chain
#extract ground-truth evidence chain and save evidence chain for training bart
python extract_evidence_chain.py --split train/dev --extract_evidence_chain --save_bart_training_data
#extract candidate evidence chain
python extract_evidence_chain.py --split train/dev --extract_candidate_evidence_chain
#evaluate ranked evidence chain
python evaluate_ranked_evidence_chain.py

#extract pretrain data
cd path_extraction
#generate inference data for bart
python parse_table_psg_link.py bart_inference_data
#generate templated pretrain data
#generate fake training and dev evidence chains
python parse_table_psg_link.py fake_pretrain_data

