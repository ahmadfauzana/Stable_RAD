import os
import time
import torch
from test import test
from train import train 
from retrieval import retrieval
# from inferences.test import inf_test
from utils import setup_seed, get_args
from setup import timer

if __name__ == '__main__':
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if args.allow_kmp_duplication:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    setup_seed(args.seed)
    
    args.score_path = os.path.join(args.score_path, 'anomaly_scores.txt')
    
    # Create the parent directory if it does not exist
    directory = os.path.dirname(args.score_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if not os.path.exists(args.score_path):
        with open(args.score_path, 'w') as file:
            file.write(f"Running in {args.phase} phase started\n")
    else:
        with open(args.score_path, 'a') as file:
            file.write(f"Running in {args.phase} phase started\n")

    print(f"Running in {args.phase} phase started")
    
    if args.phase == 'train':
        for item in args.item_list:
            start = time.time()
            train(item, args, device)
            end = time.time()
            timer(start, end, args, item)
    elif args.phase == 'test':
        for item in args.item_list:
            start = time.time()
            test(item, args, device)
            end = time.time()
            timer(start, end, args, item)
    elif args.phase == "retrieval":
        retrieval(args)

    with open(args.score_path, 'a') as file:
        file.write(f"Running in {args.phase} phase finished\n")
    print(f"Running in {args.phase} phase finished")