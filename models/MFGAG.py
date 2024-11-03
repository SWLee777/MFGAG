import torch
import tqdm
import os
from utils.utils import data2gpu, Averager, metrics, Recorder, domain_metrics
import math
from transformers import BertModel, RobertaModel
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from .layers import *
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1))

def norm(x):
    length = torch.sqrt(torch.sum(torch.pow(x, 2), dim=1)).view(-1, 1)
    x = x / length
    return x

class MFGAGModel(nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, LNN_dim, domain_num, dataset, edge_index):
        super(MFGAGModel, self).__init__()
        self.domain_num = domain_num
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.fea_size = 256
        self.emb_dim = emb_dim
        self.edge_index = edge_index
        self.gcn_in_channels = self.emb_dim + 47 * 5 + 48 if dataset == 'ch' else self.emb_dim + 38 * 5 + 32
        self.gcn_out_channels = self.emb_dim

        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large',
                                              cache_dir=r"MFGAG-main/MFGAG-main/MFGAG-main").requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        content_expert = []
        for i in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim + self.emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        emotion_expert = []
        for i in range(self.emotion_num_expert):
            emotion_expert.append(MLP(47 * 5, [256, 320], dropout, output_layer=False))
        self.emotion_expert = nn.ModuleList(emotion_expert)

        style_expert = []
        for i in range(self.style_num_expert):
            style_expert.append(MLP(48, [256, 320], dropout, output_layer=False))
        self.style_expert = nn.ModuleList(style_expert)

        self.attention = MaskAttention(emb_dim)
        self.gcn = GCNConv(in_channels=self.gcn_in_channels, out_channels=self.gcn_out_channels)

        self.expert_classifier = MLP(320 * (self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert) + self.gcn_out_channels,
                                     mlp_dims, dropout)

        # Define domain embedder
        self.domain_embedder = nn.Embedding(self.domain_num, self.emb_dim)

        # Add domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(320 * (self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, domain_num),
            nn.Softmax(dim=1)
        )

    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        category = kwargs['category']
        ids = kwargs['ids']  # Get current batch node IDs

        content_feature = self.bert(content, attention_mask=content_masks)[0]  # [batch_size ,170 ,1024]
        gate_input_feature, _ = self.attention(content_feature, content_masks)  # [batch_size, 1024]

        # Get domain embedding
        idxs = torch.tensor([index for index in category]).view(-1, 1).to(content_feature.device)  # [batch_size, 1]
        domain_embedding = self.domain_embedder(idxs).squeeze(1)  # [batch_size, 1024]

        # Add domain embedding to content expert inputs
        content_features = [expert(torch.cat([content_feature, domain_embedding.unsqueeze(1).expand_as(content_feature)], dim=2)).unsqueeze(1) for expert in self.content_expert]  # [batch_size, 1, 320]

        # Note: Domain embedding is no longer added to emotion and style features
        emotion_features = [expert(emotion_feature).unsqueeze(1) for expert in self.emotion_expert]  # [batch_size, 1, 320]
        style_features = [expert(style_feature).unsqueeze(1) for expert in self.style_expert]  # [batch_size, 1, 320]

        combined_expert_features = torch.cat(content_features + emotion_features + style_features, dim=1)
        combined_expert_features = combined_expert_features.view(combined_expert_features.size(0), -1)  # [batch_size, 320 * (semantic_num_expert + emotion_num_expert + style_num_expert)]

        # Adversarial training is applied only to expert features
        reverse_expert_features = ReverseLayerF.apply(combined_expert_features, 0.05)
        domain_pred = self.domain_discriminator(reverse_expert_features)

        # Features extracted by graph neural network
        combined_feature = torch.cat([gate_input_feature, emotion_feature, style_feature], dim=1)
        combined_feature = norm(combined_feature)
        edge_index = self.edge_index.to(combined_feature.device)
        mask = torch.isin(edge_index[0], ids) & torch.isin(edge_index[1], ids)
        batch_edge_index = edge_index[:, mask]

        id_to_idx = {id_.item(): idx for idx, id_ in enumerate(ids.cpu())}
        remapped_edge_index = torch.stack(
            [torch.tensor([id_to_idx[id_.item()] for id_ in batch_edge_index[0]], dtype=torch.int64),
             torch.tensor([id_to_idx[id_.item()] for id_ in batch_edge_index[1]], dtype=torch.int64)])

        gcn_output = self.gcn(combined_feature, remapped_edge_index.to(combined_feature.device))

        # Combine features from the graph neural network and expert outputs
        combined_task_feature = torch.cat([gcn_output, combined_expert_features], dim=1)

        deep_logits = self.expert_classifier(combined_task_feature)
        return torch.sigmoid(deep_logits.squeeze(1)), domain_pred

class Trainer():
    def __init__(self, emb_dim, mlp_dims, use_cuda, lr, dropout, dataset, train_loader, val_loader, test_loader, category_dict, weight_decay, save_param_dir, semantic_num, emotion_num, style_num, lnn_dim, edge_index, early_stop=5, epoches=100):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
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
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset
        self.edge_index = edge_index  # Add edge_index as a parameter

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

        self.best_model_path = None  # Keep track of the best model path

    def train(self, logger=None):
        if logger:
            logger.info('start training......')

        self.model = MFGAGModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
                                 self.style_num, self.lnn_dim, len(self.category_dict), self.dataset, self.edge_index)
        if self.use_cuda:
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)

        # Use ReduceLROnPlateau scheduler
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epoches}")
            avg_loss = Averager()
            avg_domain_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                domain_labels = batch_data['category']  # Use domain labels

                # Convert domain_labels to one-hot encoding
                domain_labels_one_hot = torch.nn.functional.one_hot(domain_labels,
                                                                    num_classes=len(self.category_dict)).float()

                optimizer.zero_grad()
                label_pred, domain_pred = self.model(**batch_data)
                loss = loss_fn(label_pred, label.float())
                domain_loss = loss_fn(domain_pred, domain_labels_one_hot)
                total_loss = loss + domain_loss * 0.1
                total_loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
                avg_domain_loss.add(domain_loss.item())

            print(f'Training Epoch {epoch + 1}; Loss {avg_loss.item()}; Domain Loss {avg_domain_loss.item()};')

            # Run validation
            results, domain_results = self.test(self.val_loader)

            # Update learning rate based on validation loss
            # scheduler.step(avg_loss.item())

            mark = recorder.add(results)

            # Save model
            model_save_path = os.path.join(self.save_param_dir, f'model_epoch_{epoch + 1}.pkl')
            torch.save(self.model.state_dict(), model_save_path)
            print(f'Model saved at: {model_save_path}')

            if mark == 'save':
                self.best_model_path = os.path.join(self.save_param_dir, 'parameter_MFGAG_best.pkl')
                torch.save(self.model.state_dict(), self.best_model_path)
                best_metric = results['metric']
            elif mark == 'esc':
                break

        if self.best_model_path:
            self.model.load_state_dict(torch.load(self.best_model_path))
        results, domain_results = self.test(self.test_loader)
        if logger:
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(f"test res: {results}")
        print(domain_results)
        # Ensure the best model is used for final evaluation
        return results, self.best_model_path

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        domain_pred_all = []
        domain_labels_all = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader, desc="Testing")
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']

                # Convert batch_category to one-hot encoding
                batch_domain_labels = torch.nn.functional.one_hot(batch_category,
                                                                  num_classes=len(self.category_dict)).float()

                batch_label_pred, batch_domain_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                domain_pred_all.extend(batch_domain_pred.detach().cpu().numpy().tolist())
                domain_labels_all.extend(batch_domain_labels.detach().cpu().numpy().tolist())

        metrics_result = metrics(label, pred, category, self.category_dict)
        domain_metrics_result = domain_metrics(domain_labels_all, domain_pred_all)  # Calculate domain prediction metrics
        return metrics_result, domain_metrics_result
