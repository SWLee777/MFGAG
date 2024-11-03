import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder


class EANNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout):
        super(EANNModel, self).__init__()
        # Load pre-trained BERT model and freeze its parameters
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        # Define feature kernel sizes for convolutional layers
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        # Initialize convolutional layers using the feature kernel
        self.convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        # Classifier to process extracted features
        self.classifier = MLP(mlp_input_shape, mlp_dims, dropout)
        # Domain classifier for domain adaptation
        self.domain_classifier = nn.Sequential(
            MLP(mlp_input_shape, mlp_dims, dropout, False),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dims[-1], domain_num)
        )

    def forward(self, alpha, **kwargs):
        # Extract inputs from kwargs
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        # Get BERT features
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        # Extract features using convolutional layers
        feature = self.convs(bert_feature)
        output = self.classifier(feature)  # Get output from the classifier
        reverse = ReverseLayerF.apply  # Apply the reverse gradient layer
        domain_pred = self.domain_classifier(reverse(feature, alpha))  # Get domain predictions
        return torch.sigmoid(output.squeeze(1)), domain_pred  # Return sigmoid output and domain predictions


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
        if (logger):
            logger.info('start training......')
        # Initialize the EANNModel
        self.model = EANNModel(self.emb_dim, self.mlp_dims, self.domain_num, self.dropout)
        print(self.model)
        if self.use_cuda:
            self.model = self.model.cuda()  # Move model to GPU if applicable
        loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
                pred, domain_pred = self.model(**batch_data, alpha=alpha)  # Get predictions
                loss = loss_fn(pred, label.float())  # Calculate loss for predictions
                loss_adv = F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label)  # Loss for domain predictions
                loss = loss + loss_adv  # Combine losses
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
                           os.path.join(self.save_param_dir, 'parameter_eann.pkl'))  # Save model parameters
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue

        # Load the best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_eann.pkl')))
        results = self.test(self.test_loader)  # Test the model
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_eann.pkl')

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
                batch_pred, _ = self.model(**batch_data, alpha=-1)  # Get predictions with alpha set to -1

                label.extend(batch_label.detach().cpu().numpy().tolist())  # Collect labels
                pred.extend(batch_pred.detach().cpu().numpy().tolist())  # Collect predictions
                category.extend(batch_category.detach().cpu().numpy().tolist())  # Collect categories

        return metrics(label, pred, category, self.category_dict)  # Evaluate performance metrics
