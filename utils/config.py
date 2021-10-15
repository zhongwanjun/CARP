import argparse
from ast import parse
def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str,
                        default="../data/nq-with-neg-train.txt")
    parser.add_argument("--predict_file", type=str,
                        default="../data/nq-with-neg-dev.txt")
    parser.add_argument("--num_workers", default=30, type=int)
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--basic_data_path",
                        default='/home/t-wzhong/v-wanzho/ODQA/data/',type=str)
    # model
    parser.add_argument("--model_name",
                        default="bert-base-uncased", type=str)
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_c_len", default=420, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=70, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_p_len", default=350, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--cell_trim_length", default=20, type=int,
                        help="The maximum number of tokens for each cell. Cell longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--max_q_sp_len", default=50, type=int)
    parser.add_argument("--sent-level", action="store_true")
    parser.add_argument("--rnn-retriever", action="store_true")
    parser.add_argument("--predict_batch_size", default=512,
                        type=int, help="Total batch size for predictions.")

    # multi vector scheme
    parser.add_argument("--multi-vector", type=int, default=1)
    parser.add_argument("--scheme", type=str, help="how to get the multivector, layerwise or tokenwise", default="none")
    parser.add_argument("--no_proj", action="store_true")
    parser.add_argument("--shared_encoder", action="store_true")

    # momentum
    parser.add_argument("--momentum", action="store_true")
    parser.add_argument("--init-retriever", type=str, default="")
    parser.add_argument("--k", type=int, default=38400, help="memory bank size")
    parser.add_argument("--m", type=float, default=0.999, help="momentum")


    # NQ multihop trial
    parser.add_argument("--nq-multi", action="store_true", help="train the NQ retrieval model to recover from error cases")

    return parser

args = common_args()