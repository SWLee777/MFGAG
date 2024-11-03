import os
import numpy as np
import torch
import pickle
import json
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data

def worker_init_fn(worker_id):
    np.random.seed(3407)

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return pd.DataFrame(t)  # Ensure returning a DataFrame

def df_filter(df_data, category_dict):
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data

def word2input(texts, max_len, dataset):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large',
                                                  cache_dir=r"MFGAG-main")

    token_ids = [
        tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True) for
        text in texts]
    token_ids = torch.tensor(token_ids)
    masks = (token_ids != tokenizer.pad_token_id).float()
    return token_ids, masks

class bert_data():
    def __init__(self, max_len, batch_size, category_dict, dataset, graph_path, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset_name = dataset
        self.graph_path = graph_path  # Path for graph data

    def load_data(self, path, shuffle):
        self.data = df_filter(read_pkl(path), self.category_dict)
        content = self.data['content'].to_numpy()
        comments = self.data['comments'].to_numpy()
        content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
        comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        ids = torch.tensor(self.data['id'].astype(int).to_numpy())  # Include ID

        content_token_ids, content_masks = word2input(content, self.max_len, self.dataset_name)
        comments_token_ids, comments_masks = word2input(comments, self.max_len, self.dataset_name)

        dataset = TensorDataset(
            content_token_ids,
            content_masks,
            comments_token_ids,
            comments_masks,
            content_emotion,
            comments_emotion,
            emotion_gap,
            style_feature,
            label,
            category,
            ids  # Include ID
        )
        self.dataset = dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn
        )
        return dataloader

    def collate_fn(self, batch):
        content_token_ids, content_masks, comments_token_ids, comments_masks, content_emotion, comments_emotion, emotion_gap, style_feature, label, category, ids = zip(
            *batch)
        content_token_ids = torch.stack(content_token_ids)
        content_masks = torch.stack(content_masks)
        comments_token_ids = torch.stack(comments_token_ids)
        comments_masks = torch.stack(comments_masks)
        content_emotion = torch.stack(content_emotion)
        comments_emotion = torch.stack(comments_emotion)
        emotion_gap = torch.stack(emotion_gap)
        style_feature = torch.stack(style_feature)
        label = torch.stack(label)
        category = torch.stack(category)
        ids = torch.stack(ids)  # Include ID

        # Load graph data from stored file
        with open(self.graph_path, 'rb') as f:
            graph_data = pickle.load(f)

        edge_indices = graph_data.edge_index  # Get edge indices from graph data

        return {
            'content_token_ids': content_token_ids,
            'content_masks': content_masks,
            'comments_token_ids': comments_token_ids,
            'comments_masks': comments_masks,
            'content_emotion': content_emotion,
            'comments_emotion': comments_emotion,
            'emotion_gap': emotion_gap,
            'style_feature': style_feature,
            'label': label,
            'category': category,
            'edge_indices': edge_indices,  # Use stored graph data
            'ids': ids  # Include ID
        }
