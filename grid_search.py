import torch
import tqdm
import pickle
import logging
import os
import time
import json
from copy import deepcopy

from utils.dataloader import bert_data  # Import BERT data loader
from models.MFGAG import Trainer as MFGAGTrainer  # Import Trainer class from MFGAG model

# Generator function for a range of floats
def frange(x, y, jump):
    while x < y:
        x = round(x, 8)  # Keep eight decimal places
        yield x
        x += jump

# Run class definition
class Run():
    def __init__(self, config):
        # Initialize configuration parameters
        self.configinfo = config

        self.use_cuda = config['use_cuda']  # Use CUDA or not
        self.model_name = config['model_name']  # Model name
        self.batchsize = config['batchsize']  # Batch size
        self.emb_dim = config['emb_dim']  # Embedding dimension
        self.weight_decay = config['weight_decay']  # Weight decay
        self.lr = config['lr']  # Learning rate
        self.epoch = config['epoch']  # Number of training epochs
        self.max_len = config['max_len']  # Maximum sequence length
        self.num_workers = config['num_workers']  # Number of worker threads
        self.early_stop = config['early_stop']  # Early stopping strategy
        self.root_path = config['root_path']  # Dataset root path
        self.mlp_dims = config['model']['mlp']['dims']  # MLP layer dimensions
        self.dropout = config['model']['mlp']['dropout']  # Dropout probability
        self.seed = config['seed']  # Random seed
        self.save_log_dir = config['save_log_dir']  # Log saving directory
        self.save_param_dir = config['save_param_dir']  # Model parameter saving directory
        self.param_log_dir = config['param_log_dir']  # Parameter log directory

        self.semantic_num = config['semantic_num']  # Number of semantic experts
        self.emotion_num = config['emotion_num']  # Number of emotion experts
        self.style_num = config['style_num']  # Number of style experts
        self.lnn_dim = config['lnn_dim']  # Interaction representation dimension
        self.domain_num = config['domain_num']  # Number of domains
        self.category_dict = config['category_dict']  # Category dictionary
        self.dataset = config['dataset']  # Dataset name

        self.train_path = self.root_path + 'train_id_325.pkl'  # Training set path
        self.val_path = self.root_path + 'val_id_325.pkl'  # Validation set path
        self.test_path = self.root_path + 'test_id_325.pkl'  # Test set path

        # Path for global graph data
        self.global_graph_path = config['global_graph_path']

    # Get data loaders
    def get_dataloader(self):
        loader = bert_data(max_len=self.max_len, batch_size=self.batchsize,
                           category_dict=self.category_dict, num_workers=self.num_workers, dataset=self.dataset,
                           graph_path=self.global_graph_path)  # Pass global graph data path
        train_loader = loader.load_data(self.train_path, True)  # Load training data
        val_loader = loader.load_data(self.val_path, False)  # Load validation data
        test_loader = loader.load_data(self.test_path, False)  # Load test data

        return train_loader, val_loader, test_loader  # Return dataloaders

    # Get file logger
    def getFileLogger(self, log_file):
        logger = logging.getLogger()  # Get logger
        logger.setLevel(level=logging.INFO)  # Set log level to INFO
        handler = logging.FileHandler(log_file)  # Create file handler
        handler.setLevel(logging.INFO)  # Set handler log level to INFO
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Set log format
        handler.setFormatter(formatter)  # Apply format to handler
        logger.addHandler(handler)  # Add handler to logger
        return logger

    # Convert configuration parameters to dictionary
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    # Main function
    def main(self):
        param_log_dir = self.param_log_dir
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)  # Create parameter log directory
        param_log_file = os.path.join(param_log_dir, self.model_name + '_' + 'oneloss_param.txt')
        logger = self.getFileLogger(param_log_file)  # Get logger

        train_loader, val_loader, test_loader = self.get_dataloader()  # Get data loaders

        # Load global graph data (2, 14055668)
        with open(self.global_graph_path, 'rb') as f:
            global_graph = pickle.load(f)
        edge_index = global_graph.edge_index

        # Keep only the trainer of MFGAG model
        trainer = MFGAGTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, dataset=self.dataset,
                                use_cuda=self.use_cuda,
                                lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                weight_decay=self.weight_decay,
                                val_loader=val_loader, test_loader=test_loader, category_dict=self.category_dict,
                                early_stop=self.early_stop, epoches=self.epoch,
                                save_param_dir=os.path.join(self.save_param_dir, self.model_name),
                                semantic_num=self.semantic_num, emotion_num=self.emotion_num, style_num=self.style_num,
                                lnn_dim=self.lnn_dim, edge_index=edge_index)

        train_param = {
            'lr': [self.lr] * 10  # Training parameters, containing ten identical learning rates
        }
        print(train_param)

        param = train_param
        best_param = []
        json_path = './logs/json/' + self.model_name + '.json'
        json_result = []
        train_once = True  # Variable to control training times

        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                setattr(trainer, p, v)
                metrics, model_path = trainer.train(logger)
                json_result.append(metrics)
                if (metrics['metric'] > best_metric['metric']):
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
                if train_once:  # Check variable to decide whether to continue training
                    break
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('--------------------------------------\n')

        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)
