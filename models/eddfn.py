import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder


class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))  # Add linear layer
            layers.append(torch.nn.ReLU())  # Add ReLU activation
            layers.append(torch.nn.Dropout(p=dropout))  # Add dropout layer
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))  # Output layer
        self.mlp = torch.nn.Sequential(*layers)  # Create sequential model

    def forward(self, x):
        """
        Forward pass of the MLP.
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)  # Pass input through MLP


class EDDFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout):
        super(EDDFNModel, self).__init__()
        # Load pre-trained BERT model and freeze its parameters
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        # Define shared and specific MLPs
        self.shared_mlp = MLP(emb_dim, mlp_dims, dropout, False)
        self.specific_mlp = torch.nn.ModuleList([MLP(emb_dim, mlp_dims, dropout, False) for i in range(9)])
        self.decoder = MLP(mlp_dims[-1] * 2, (64, emb_dim), dropout, False)  # Decoder for reconstruction
        self.classifier = torch.nn.Linear(2 * mlp_dims[-1], 1)  # Classifier
        self.domain_classifier = nn.Sequential(
            MLP(mlp_dims[-1], mlp_dims, dropout, False),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dims[-1], domain_num)  # Domain classifier
        )
        self.attention = MaskAttention(emb_dim)  # Attention mechanism

    def forward(self, alpha=1, **kwargs):
        # Extract inputs from kwargs
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        # Get BERT features
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        bert_feature, _ = self.attention(bert_feature, masks)  # Apply attention

        specific_feature = []
        for i in range(bert_feature.size(0)):
            specific_feature.append(
                self.specific_mlp[category[i]](bert_feature[i].view(1, -1)))  # Get specific features

        specific_feature = torch.cat(specific_feature)  # Concatenate specific features
        shared_feature = self.shared_mlp(bert_feature)  # Get shared features
        feature = torch.cat([shared_feature, specific_feature], 1)  # Combine features
        rec_feature = self.decoder(feature)  # Decode features
        output = self.classifier(feature)  # Get output

        reverse = ReverseLayerF.apply  # Apply reverse gradient layer
        domain_pred = self.domain_classifier(reverse(shared_feature, alpha))  # Get domain predictions

        return torch.sigmoid(output.squeeze(1)), rec_feature, bert_feature, domain_pred  # Return outputs


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 domain_num,
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.domain_num = domain_num

        # Create directory to save parameters if it does not exist
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self, logger=None):
        print(self.save_param_dir)
        if (logger):
            logger.info("start training......")
        # Initialize the EDDFNModel
        self.model = EDDFNModel(self.emb_dim, self.mlp_dims, self.domain_num, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()  # Move model to GPU if applicable
        loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy Loss
        loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)  # Mean Squared Error Loss
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)  # For tracking early stopping
        for epoch in range(self.epoches):
            self.model.train()  # Set model to training mode
            train_data_iter = tqdm.tqdm(self.train_loader)  # Progress bar for training
            avg_loss = Averager()  # To average the loss
            # Update alpha for gradient reversal
            alpha = max(2. / (1. + np.exp(-10 * epoch / self.epoches)) - 1, 1e-1)

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)  # Move data to GPU if applicable
                label = batch_data['label']
                domain_label = batch_data['category']

                optimizer.zero_grad()  # Reset gradients
                # Get predictions and features
                pred, rec_feature, bert_feature, domain_pred = self.model(**batch_data, alpha=alpha)
                # Calculate combined loss
                loss = loss_fn(pred, label.float()) + loss_mse(rec_feature, bert_feature) + 0.1 * F.nll_loss(
                    F.log_softmax(domain_pred, dim=1), domain_label)
                optimizer.zero_grad()  # Reset gradients again (redundant)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                avg_loss.add(loss.item())  # Add to average loss

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = "[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}".format(epoch, str(self.lr), loss.item(),
                                                                                   avg_loss)

            results = self.test(self.val_loader)  # Validate the model
            mark = recorder.add(results)  # Check if we should save or early stop
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_eddfn.pkl'))  # Save model parameters
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue

        # Load the best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_eddfn.pkl')))
        results = self.test(self.test_loader)  # Test the model
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_eddfn.pkl')

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
                batch_pred, _, __, ___ = self.model(**batch_data)  # Get predictions

                label.extend(batch_label.detach().cpu().numpy().tolist())  # Collect labels
                pred.extend(batch_pred.detach().cpu().numpy().tolist())  # Collect predictions
                category.extend(batch_category.detach().cpu().numpy().tolist())  # Collect categories

        return metrics(label, pred, category, self.category_dict)  # Evaluate performance metrics
