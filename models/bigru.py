import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import logging
from torch.optim import AdamW


class BiGRUModel(torch.nn.Module):
    def __init__(self, emb_dim, num_layers, mlp_dims, dropout):
        super(BiGRUModel, self).__init__()
        self.fea_size = emb_dim  # Feature size
        # Load pre-trained BERT model and freeze its parameters
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.embedding = self.bert.embeddings  # Extract the embedding layer

        # Initialize a bidirectional GRU
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)

        input_shape = self.fea_size * 2  # Output size from bidirectional GRU
        self.attention = MaskAttention(input_shape)  # Initialize attention mechanism
        self.mlp = MLP(input_shape, mlp_dims, dropout)  # Initialize MLP

    def forward(self, **kwargs):
        # Extract inputs and masks from kwargs
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        # Get feature embeddings
        feature = self.embedding(inputs)
        # Pass through GRU
        feature, _ = self.rnn(feature)
        # Apply attention mechanism
        feature, _ = self.attention(feature, masks)
        # Pass through MLP and apply sigmoid activation
        output = self.mlp(feature)
        return torch.sigmoid(output.squeeze(1))


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 num_layers,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.num_layers = num_layers

        # Create directory to save parameters if it does not exist
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self, logger=None):
        if (logger):
            logger.info('start training......')
        # Initialize the model
        self.model = BiGRUModel(self.emb_dim, self.num_layers, self.mlp_dims, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)  # For tracking early stopping
        best_metric = recorder.cur['metric']
        for epoch in range(self.epoches):
            self.model.train()  # Set model to training mode
            train_data_iter = tqdm.tqdm(self.train_loader)  # Progress bar for training
            avg_loss = Averager()  # To average the loss

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)  # Move data to GPU if applicable
                label = batch_data['label']

                optimizer.zero_grad()  # Reset gradients
                pred = self.model(**batch_data)  # Get predictions
                loss = loss_fn(pred, label.float())  # Calculate loss
                optimizer.zero_grad()  # Reset gradients again (redundant)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                avg_loss.add(loss.item())  # Add to average loss
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(),
                                                                                   avg_loss)

            results = self.test(self.val_loader)  # Validate the model
            mark = recorder.add(results)  # Check if we should save or early stop
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_bigru.pkl'))  # Save model parameters
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(
            torch.load(os.path.join(self.save_param_dir, 'parameter_bigru.pkl')))  # Load the best model
        results = self.test(self.test_loader)  # Test the model
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_bigru.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()  # Set model to evaluation mode
        data_iter = tqdm.tqdm(dataloader)  # Progress bar for testing
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():  # Disable gradient calculation
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())  # Collect labels
                pred.extend(batch_pred.detach().cpu().numpy().tolist())  # Collect predictions
                category.extend(batch_category.detach().cpu().numpy().tolist())  # Collect categories

        return metrics(label, pred, category, self.category_dict)  # Evaluate performance metrics
