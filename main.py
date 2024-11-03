import os
import argparse
import time
import torch
import numpy as np
import random

from grid_search import Run

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='MFGAG')  # Model name parameter, default is 'MFGAG'
parser.add_argument('--epoch', type=int, default=100)  # Number of training epochs, default is 100
parser.add_argument('--max_len', type=int, default=340)  # Maximum sequence length, default is 340
parser.add_argument('--num_workers', type=int, default=4)  # Number of worker threads, default is 4
parser.add_argument('--early_stop', type=int, default=3)  # Early stopping strategy, default is 3
parser.add_argument('--dataset', default='ch')  # Dataset parameter, default is 'ch'
parser.add_argument('--batchsize', type=int, default=64)  # Batch size parameter, default is 64
parser.add_argument('--seed', type=int, default=3407)  # Random seed parameter, default is 3407
parser.add_argument('--gpu', default='0')  # GPU device parameter, default is '0'
parser.add_argument('--emb_dim', type=int, default=1024)  # Embedding dimension parameter, default is 1024
parser.add_argument('--lr', type=float, default=0.0004)  # Learning rate parameter, default is 0.0004
parser.add_argument('--save_log_dir', default='./logs')  # Log saving directory parameter, default is './logs'
parser.add_argument('--save_param_dir', default='./param_model')  # Model parameter saving directory parameter, default is './param_model'
parser.add_argument('--param_log_dir', default='./logs/param')  # Parameter log directory parameter, default is './logs/param'
parser.add_argument('--semantic_num', type=int, default=7)  # Number of semantic experts, default is 7
parser.add_argument('--emotion_num', type=int, default=7)  # Number of emotion experts, default is 7
parser.add_argument('--style_num', type=int, default=2)  # Number of style experts, default is 2
parser.add_argument('--lnn_dim', type=int, default=20)  # Interaction representation dimension, default is 20
parser.add_argument('--domain_num', type=int, default=9)  # Number of domains, default is 9
parser.add_argument('--global_graph_path', default='./data/ch/global_edges_325.pkl')  # Global graph data path

# Parse command line arguments
args = parser.parse_args()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set random seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Choose root path and category dictionary based on dataset
root_path = r'./data/ch/'
if args.domain_num == 9:
    category_dict = {
        "科技": 0,
        "军事": 1,
        "教育考试": 2,
        "灾难事故": 3,
        "政治": 4,
        "医药健康": 5,
        "财经商业": 6,
        "文体娱乐": 7,
        "社会生活": 8,
    }
elif args.domain_num == 6:
    category_dict = {
        "教育考试": 0,
        "灾难事故": 1,
        "医药健康": 2,
        "财经商业": 3,
        "文体娱乐": 4,
        "社会生活": 5,
    }
elif args.domain_num == 3:
    category_dict = {
        "政治": 0,
        "医药健康": 1,
        "文体娱乐": 2,
    }

# Print configuration information
print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}; domain_num: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu, args.domain_num))

# Configuration dictionary to pass to the Run instance
config = {
    'use_cuda': True,  # Use CUDA
    'batchsize': args.batchsize,  # Batch size
    'max_len': args.max_len,  # Maximum sequence length
    'early_stop': args.early_stop,  # Early stopping strategy
    'num_workers': args.num_workers,  # Number of worker threads
    'root_path': root_path,  # Dataset root path
    'weight_decay': 5e-5,  # Weight decay
    'category_dict': category_dict,  # Category dictionary
    'dataset': args.dataset,  # Dataset
    'model': {
        'mlp': {'dims': [384], 'dropout': 0.2}  # Model configuration, MLP layer parameters
    },
    'emb_dim': args.emb_dim,  # Embedding dimension
    'lr': args.lr,  # Learning rate
    'epoch': args.epoch,  # Number of training epochs
    'model_name': args.model_name,  # Model name
    'seed': args.seed,  # Random seed
    'semantic_num': args.semantic_num,  # Number of semantic experts
    'emotion_num': args.emotion_num,  # Number of emotion experts
    'style_num': args.style_num,  # Number of style experts
    'domain_num': args.domain_num,  # Number of domains
    'lnn_dim': args.lnn_dim,  # Interaction representation dimension
    'save_log_dir': args.save_log_dir,  # Log saving directory
    'save_param_dir': args.save_param_dir,  # Model parameter saving directory
    'param_log_dir': args.param_log_dir,  # Parameter log directory
    'global_graph_path': args.global_graph_path  # Global graph data path
}

if __name__ == '__main__':
    # Create and run instance
    run_instance = Run(config=config)
    results = run_instance.main()
